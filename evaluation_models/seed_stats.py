"""
Variabilidad de los resultados: varias semillas de entrenamiento + bootstrap por bloques sobre test.

Parte A (todas las métricas de evaluation.py): media ± desviación típica entre semillas, leyendo el
metrics.pkl de cada ejecución. -> seeds_metrics_summary.txt / .csv

Parte B (métricas principales): bootstrap por bloques.
  - Test se remuestrea por bloques de --block canales consecutivos (con reemplazo), porque los canales
    consecutivos están correlados; los canales generados (i.i.d.) se remuestrean canal a canal.
  - En cada réplica se calculan las métricas de cada modelo y semilla frente al test remuestreado y se
    promedian entre semillas. Del conjunto de réplicas salen:
      * IC 95 % (percentiles 2.5 y 97.5) de la media entre semillas,
      * P(< Gaussian): fracción de réplicas en que el modelo mejora al modelo gaussiano (misma réplica de
        test para ambos: comparación emparejada).
  - La referencia "real contra real" (train y val frente a test) se trata igual: train y val también se
    remuestrean por bloques.
  -> bootstrap_ci.txt / .csv

Nota: KS y |Δlog10 BER| son estadísticos de error absoluto y su distribución bootstrap está sesgada hacia
arriba (al remuestrear, el ruido nunca reduce una distancia de cero): el valor puntual puede quedar cerca
del extremo inferior del IC o incluso fuera. Los IC indican sobre todo la dispersión; la comparación
entre fuentes debe hacerse con P(< Gaussian), que es emparejada.

Todas las métricas son "menor es mejor" (KS, MAE, |Δlog10 BER|). Para que el bootstrap sea rápido, la
KS de |h| se calcula con todos los taps sobre una rejilla de 0.1 dB (evaluation.py usa una submuestra de
50 000 valores) y la BER de cada conjunto es la media de la BER por canal de **todos** sus canales
(evaluation.py usa 900 por conjunto); por eso los valores puntuales difieren ligeramente de los de
metrics_summary.txt.

Uso (desde la raíz del repositorio):
    python evaluation_models/seed_stats.py
    python evaluation_models/seed_stats.py --runs eval_results/X eval_results/X/seeds/seed1 --n-boot 1000
"""
import argparse
import csv
import glob
import os
import pickle
import sys

import numpy as np
from scipy.stats import ks_2samp

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
import evaluation as ev  # noqa: E402
from BER import BITS_PER_SYMBOL, channel_frequency_response, demodulate, modulate  # noqa: E402

GEN_MODELS = ["gan", "dcgan", "wgan", "ddpm", "ddim"]
CLASSIC = ["tdl", "gauss"]
REF = "gauss"


# ---------------------------------------------------------------- Parte A

def seed_table(runs, save_dir):
    mets = []
    for r in runs:
        with open(os.path.join(r, "metrics.pkl"), "rb") as f:
            mets.append(pickle.load(f)["metrics"])
    models = [m for m in GEN_MODELS + CLASSIC if all(m in x for x in mets)]
    keys = [k for k in mets[0][models[0]] if all(np.isscalar(x[m].get(k)) for x in mets for m in models)]
    rows = []
    for k in keys:
        row = {"metric": k}
        for m in models:
            v = np.array([x[m][k] for x in mets], dtype=float)
            row[f"{m}_mean"], row[f"{m}_std"] = float(np.nanmean(v)), float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0
        rows.append(row)
    with open(os.path.join(save_dir, "seeds_metrics_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = [f"Media ± desviación típica entre {len(runs)} semillas de entrenamiento (referencia: test)",
             "Ejecuciones: " + ", ".join(runs), "",
             f"{'metric':<30}" + "".join(f"{ev.MODEL_LABELS[m]:>22}" for m in models)]
    for r in rows:
        cells = []
        for m in models:
            mu, sd = r[f"{m}_mean"], r[f"{m}_std"]
            fmt = ".3e" if 0 < abs(mu) < 1e-2 else ".4g"
            cells.append(f"{format(mu, fmt)} ± {format(sd, '.2g')}")
        lines.append(f"{r['metric']:<30}" + "".join(f"{c:>22}" for c in cells))
    with open(os.path.join(save_dir, "seeds_metrics_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Parte A: {len(keys)} métricas, {len(models)} modelos, {len(runs)} semillas")


# ---------------------------------------------------------------- Parte B

def per_channel_ber(h, snr_db, mod, seed):
    """BER de cada canal (N, S) con el mismo esquema que BER.ber_curve (ZF, mismos bits/ruido por canal)."""
    H = channel_frequency_response(h)
    N, K = H.shape
    bps = BITS_PER_SYMBOL[mod]
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=N * K * bps, dtype=np.int8)
    x = modulate(bits, mod).reshape(1, N, K)
    w = (rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape)) / np.sqrt(2)
    Hs = np.where(np.abs(H) > 1e-15, H, 1e-15)[None]
    out = np.zeros((N, len(snr_db)))
    for j, snr in enumerate(snr_db):
        y = H[None] * x + np.sqrt(10 ** (-snr / 10)) * w
        rx = demodulate((y / Hs).reshape(-1), mod)
        out[:, j] = (rx != bits).reshape(N, K * bps).mean(axis=1)
    return out


class Source:
    """Rasgos por canal de un conjunto de canales, para remuestrear rápido."""

    def __init__(self, h, dt, cfg, mag_edges, ber_cfg, seed, n_ber):
        ev_cfg = cfg["evaluation"]
        g = np.linspace(0, 1, 2)
        st = ev.channel_stats(h, dt, g, g, kde_points=10, seed=seed,
                              delay_thresholds=ev_cfg["delay_thresholds_db"],
                              channel_params=ev_cfg.get("channel_params"))
        self.n = len(h)
        self.p = np.abs(h) ** 2
        self.feat = {k[:-4]: v for k, v in st.items() if k.endswith("_all")}
        mag_db = 20 * np.log10(np.abs(h) + 1e-12)
        idx = np.clip(np.searchsorted(mag_edges, mag_db) - 1, 0, len(mag_edges) - 2)
        self.mag_hist = np.zeros((self.n, len(mag_edges) - 1))
        np.add.at(self.mag_hist, (np.repeat(np.arange(self.n), h.shape[1]), idx.ravel()), 1)
        nb = min(n_ber, self.n)
        self.ber = {mod: per_channel_ber(h[:nb], ber_cfg["snr_db"], mod, seed) for mod in ber_cfg["modulations"]}


def metric_values(src, idx, ref, ridx, mask, feat_keys):
    """Métricas de src[idx] frente a ref[ridx]. Devuelve dict."""
    out = {}
    pdp_f = 10 * np.log10(src.p[idx].mean(0) + 1e-30)
    pdp_r = 10 * np.log10(ref.p[ridx].mean(0) + 1e-30)
    out["pdp_mae_db"] = float(np.mean(np.abs(pdp_f - pdp_r)[mask]))
    cf = np.cumsum(src.mag_hist[idx].sum(0))
    cr = np.cumsum(ref.mag_hist[ridx].sum(0))
    out["mag_ks"] = float(np.max(np.abs(cf / cf[-1] - cr / cr[-1])))
    for k in feat_keys:
        out[f"{k}_ks"] = float(ks_2samp(src.feat[k][idx], ref.feat[k][ridx]).statistic)
    for mod in src.ber:
        nb_s, nb_r = len(src.ber[mod]), len(ref.ber[mod])
        bs = src.ber[mod][idx[idx < nb_s]].mean(0) if np.any(idx < nb_s) else src.ber[mod].mean(0)
        br = ref.ber[mod][ridx[ridx < nb_r]].mean(0)
        out[f"ber_{mod}_dlog10"] = float(np.mean(np.abs(np.log10(np.maximum(bs, 1e-12))
                                                        - np.log10(np.maximum(br, 1e-12)))))
    return out


def block_resample(n, block, rng):
    starts = np.arange(0, n, block)
    pick = rng.choice(starts, size=len(starts), replace=True)
    return np.concatenate([np.arange(s, min(s + block, n)) for s in pick])


def main():
    parser = argparse.ArgumentParser(description="Semillas + bootstrap por bloques")
    parser.add_argument("--config", default=os.path.join(THIS_DIR, "evaluation_config.json"))
    parser.add_argument("--runs", nargs="+", default=None,
                        help="Carpetas de evaluación (una por semilla). Por defecto paths.save_dir y "
                             "paths.save_dir/seeds/seed*")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--block", type=int, default=50, help="Tamaño de bloque del bootstrap de test")
    parser.add_argument("--out", default=None, help="Carpeta de salida (por defecto paths.save_dir)")
    args = parser.parse_args()

    cfg = ev.load_config(args.config)
    base = ev.repo_path(cfg["paths"]["save_dir"])
    runs = args.runs or [base] + sorted(glob.glob(os.path.join(base, "seeds", "seed*")))
    runs = [ev.repo_path(r) for r in runs if os.path.exists(os.path.join(ev.repo_path(r), "metrics.pkl"))]
    out_dir = ev.repo_path(args.out) if args.out else base
    print(f"{len(runs)} ejecuciones:", *runs, sep="\n  ")
    seed_table(runs, out_dir)

    # Datos reales
    data = np.load(ev.repo_path(cfg["data"]["path"])).astype(np.float32)
    split = np.load(ev.repo_path(cfg["data"]["split_file"]))
    real = {k: ev.to_complex(data[np.sort(split[k])]) for k in ("test", "train", "val")}
    dt = ev.sampling_period(cfg)
    seed = cfg["experiment"]["seed"]
    ber_cfg = {"snr_db": cfg["ber"]["snr_db"], "modulations": cfg["ber"]["modulations"]}
    n_ber = 10 ** 9  # BER por canal de todos los canales de cada conjunto
    # Rejilla de |h| en dB (la KS no cambia con una transformación monótona; en dB la rejilla resuelve bien
    # los valores pequeños, que son la mayoría)
    mag_edges = np.concatenate([[-1e9], np.linspace(-120, 20, 1401), [1e9]])
    feat_keys = None

    print("Rasgos por canal ...", flush=True)
    srcs = {}
    for k, h in real.items():
        srcs[("real", k)] = Source(h, dt, cfg, mag_edges, ber_cfg, seed, n_ber)
    for r in runs:
        for m in GEN_MODELS + CLASSIC:
            p = os.path.join(r, "generated", f"h_generated_{m}.npy")
            if os.path.exists(p):
                srcs[(m, r)] = Source(ev.to_complex(np.load(p)), dt, cfg, mag_edges, ber_cfg, seed, n_ber)
    test = srcs[("real", "test")]
    feat_keys = [k for k in test.feat if not k.startswith("_")]
    mask = test.p.mean(0) > test.p.mean(0).max() * 1e-6
    models = [m for m in GEN_MODELS + CLASSIC if all((m, r) in srcs for r in runs)]
    print(f"Modelos con todas las semillas: {models}", flush=True)

    def all_metrics(rng):
        ridx = block_resample(test.n, args.block, rng) if rng is not None else np.arange(test.n)
        res = {}
        for k in ("train", "val"):
            s = srcs[("real", k)]
            idx = block_resample(s.n, args.block, rng) if rng is not None else np.arange(s.n)
            res[f"real_{k}"] = metric_values(s, idx, test, ridx, mask, feat_keys)
        for m in models:
            vals = []
            for r in runs:
                s = srcs[(m, r)]
                idx = rng.integers(0, s.n, s.n) if rng is not None else np.arange(s.n)
                vals.append(metric_values(s, idx, test, ridx, mask, feat_keys))
            res[m] = {k: np.array([v[k] for v in vals]) for k in vals[0]}
        return res

    point = all_metrics(None)
    rng = np.random.default_rng(seed)
    keys = list(point["real_train"])
    names = ["real_train", "real_val"] + models
    boot = {n: {k: np.zeros(args.n_boot) for k in keys} for n in names}
    for b in range(args.n_boot):
        res = all_metrics(rng)
        for n in names:
            for k in keys:
                boot[n][k][b] = np.mean(res[n][k])
        if (b + 1) % 100 == 0:
            print(f"  réplica {b + 1}/{args.n_boot}", flush=True)

    rows = []
    for k in keys:
        for n in names:
            pv = point[n][k]
            row = {"metric": k, "source": n,
                   "mean": float(np.mean(pv)), "seed_std": float(np.std(pv, ddof=1)) if np.size(pv) > 1 else 0.0,
                   "ci_low": float(np.percentile(boot[n][k], 2.5)), "ci_high": float(np.percentile(boot[n][k], 97.5)),
                   "p_better_than_gauss": float(np.mean(boot[n][k] < boot[REF][k])) if REF in boot and n != REF
                   else np.nan}
            rows.append(row)
    with open(os.path.join(out_dir, "bootstrap_ci.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    label = {"real_train": "Real train", "real_val": "Real val", **ev.MODEL_LABELS}
    lines = [f"Bootstrap por bloques ({args.n_boot} réplicas; test en bloques de {args.block} canales "
             f"consecutivos, generados canal a canal) y {len(runs)} semillas de entrenamiento.",
             "mean ± std: media y desviación típica entre semillas (datos completos). IC95: intervalo del "
             "bootstrap de la media entre semillas.",
             "P(<Gauss): fracción de réplicas en que la fuente mejora al gaussiano (emparejada; > 0.975 o "
             "< 0.025 = diferencia significativa al 5 %). Menor es mejor en todas las métricas.",
             "Referencia real: train y val (también remuestreados por bloques) frente a test.", ""]
    for k in keys:
        lines.append(f"== {k}")
        for r in [x for x in rows if x["metric"] == k]:
            pb = "" if np.isnan(r["p_better_than_gauss"]) else f"{r['p_better_than_gauss']:.3f}"
            lines.append(f"  {label[r['source']]:<12} {r['mean']:>9.4f} ± {r['seed_std']:<8.4f} "
                         f"IC95 [{r['ci_low']:.4f}, {r['ci_high']:.4f}]   P(<Gauss) {pb}")
    with open(os.path.join(out_dir, "bootstrap_ci.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Resultados en {out_dir}/bootstrap_ci.txt y seeds_metrics_summary.txt")


if __name__ == "__main__":
    main()
