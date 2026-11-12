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
  - --block segment: un bloque por tramo continuo entre discontinuidades (recordings.py).
  -> bootstrap_ci.txt / .csv (bloque 50) o bootstrap_ci_block<valor>.txt / .csv (otros bloques)

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
import multiprocessing as mp
import os
import pickle
import sys

import numpy as np
import torch
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


def block_resample(n, block, rng, labels=None):
    """Índices remuestreados por bloques. block: tamaño fijo (canales consecutivos) o, si labels no es
    None, un bloque por cada valor distinto de labels (tramo continuo de recordings.py); se sortean tantos
    tramos como hay, con reemplazo (el número de canales de la réplica varía)."""
    if labels is not None:
        groups = [np.where(labels == u)[0] for u in np.unique(labels)]
        pick = rng.integers(0, len(groups), len(groups))
        return np.concatenate([groups[j] for j in pick])
    starts = np.arange(0, n, block)
    pick = rng.choice(starts, size=len(starts), replace=True)
    return np.concatenate([np.arange(s, min(s + block, n)) for s in pick])


_REPLICATE = None


def _run_replicate(b):
    return b, _REPLICATE(b)


def main():
    parser = argparse.ArgumentParser(description="Semillas + bootstrap por bloques")
    parser.add_argument("--config", default=os.path.join(THIS_DIR, "evaluation_config.json"))
    parser.add_argument("--runs", nargs="+", default=None,
                        help="Carpetas de evaluación (una por semilla). Por defecto paths.save_dir y "
                             "paths.save_dir/seeds/seed*")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--block", default="50",
                        help="Tamaño de bloque del bootstrap (canales consecutivos), o 'segment' para un bloque "
                             "por tramo continuo (recordings.npz de recordings.py en paths.save_dir). Con un "
                             "valor distinto de 50 los ficheros de salida llevan el sufijo _block<valor>")
    parser.add_argument("--out", default=None, help="Carpeta de salida (por defecto paths.save_dir)")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                        help="Procesos para las réplicas del bootstrap")
    parser.add_argument("--cache", default=None,
                        help="Carpeta para guardar rasgos por canal y réplicas parciales (permite reanudar)")
    args = parser.parse_args()

    cfg = ev.load_config(args.config)
    base = ev.repo_path(cfg["paths"]["save_dir"])
    runs = args.runs or [base] + sorted(glob.glob(os.path.join(base, "seeds", "seed*")))
    runs = [ev.repo_path(r) for r in runs if os.path.exists(os.path.join(ev.repo_path(r), "metrics.pkl"))]
    out_dir = ev.repo_path(args.out) if args.out else base
    by_segment = args.block == "segment"
    block = None if by_segment else int(args.block)
    suffix = "" if args.block == "50" else f"_block{args.block}"
    os.makedirs(out_dir, exist_ok=True)
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
    cache = ev.repo_path(args.cache) if args.cache else None
    if cache:
        os.makedirs(cache, exist_ok=True)

    def get_source(key, loader):
        # Los rasgos por canal son lo más caro; con --cache se guardan para reanudar
        path = os.path.join(cache, "src_" + "_".join(os.path.basename(str(x)) for x in key) + ".pkl") \
            if cache else None
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        src = Source(loader(), dt, cfg, mag_edges, ber_cfg, seed, n_ber)
        if path:
            with open(path, "wb") as f:
                pickle.dump(src, f)
        print(f"  {key}", flush=True)
        return src

    srcs = {}
    for k, h in real.items():
        srcs[("real", k)] = get_source(("real", k), lambda h=h: h)
    for r in runs:
        for m in GEN_MODELS + CLASSIC:
            p = os.path.join(r, "generated", f"h_generated_{m}.npy")
            if os.path.exists(p):
                srcs[(m, r)] = get_source((m, r), lambda p=p: ev.to_complex(np.load(p)))
    test = srcs[("real", "test")]
    labels = {k: None for k in real}
    if by_segment:
        seg_f = np.load(os.path.join(base, "recordings.npz"))["segment_filtered"]
        labels = {k: seg_f[np.sort(split[k])] for k in real}
        print("Tramos por conjunto: " + ", ".join(f"{k} {len(np.unique(v))}" for k, v in labels.items()), flush=True)
    feat_keys = [k for k in test.feat if not k.startswith("_")]
    mask = test.p.mean(0) > test.p.mean(0).max() * 1e-6
    models = [m for m in GEN_MODELS + CLASSIC if all((m, r) in srcs for r in runs)]
    print(f"Modelos con todas las semillas: {models}", flush=True)

    def all_metrics(rng):
        ridx = block_resample(test.n, block, rng, labels["test"]) if rng is not None else np.arange(test.n)
        res = {}
        for k in ("train", "val"):
            s = srcs[("real", k)]
            idx = block_resample(s.n, block, rng, labels[k]) if rng is not None else np.arange(s.n)
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
    keys = list(point["real_train"])
    names = ["real_train", "real_val"] + models
    global _REPLICATE
    _REPLICATE = lambda b: {n: {k: float(np.mean(v)) for k, v in r.items()}  # noqa: E731
                            for n, r in all_metrics(np.random.default_rng([seed, b])).items()}
    # Cada réplica usa su propio generador ([seed, b]): el resultado no depende del orden ni del número de
    # procesos, y las réplicas ya hechas (guardadas en --cache) se pueden reutilizar al reanudar
    done = {}
    part = os.path.join(cache, f"replicates{suffix}.pkl") if cache else None
    if part and os.path.exists(part):
        with open(part, "rb") as f:
            done = pickle.load(f)
    todo = [b for b in range(args.n_boot) if b not in done]
    print(f"Réplicas: {len(done)} hechas, {len(todo)} pendientes ({args.workers} procesos)", flush=True)
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers, initializer=torch.set_num_threads, initargs=(1,)) as pool:
        for i, (b, res) in enumerate(pool.imap_unordered(_run_replicate, todo, chunksize=4)):
            done[b] = res
            if (i + 1) % 50 == 0 or i + 1 == len(todo):
                print(f"  réplica {len(done)}/{args.n_boot}", flush=True)
                if part:
                    with open(part + ".tmp", "wb") as f:
                        pickle.dump(done, f)
                    os.replace(part + ".tmp", part)
    boot = {n: {k: np.array([done[b][n][k] for b in range(args.n_boot)]) for k in keys} for n in names}

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
    with open(os.path.join(out_dir, f"bootstrap_ci{suffix}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    label = {"real_train": "Real train", "real_val": "Real val", **ev.MODEL_LABELS}
    blk = ("un bloque por tramo continuo (recordings.py)" if by_segment
           else f"bloques de {block} canales consecutivos")
    lines = [f"Bootstrap por bloques ({args.n_boot} réplicas; test, train y val en {blk}; "
             f"generados canal a canal) y {len(runs)} semillas de entrenamiento.",
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
    with open(os.path.join(out_dir, f"bootstrap_ci{suffix}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Resultados en {out_dir}/bootstrap_ci{suffix}.txt y seeds_metrics_summary.txt")


if __name__ == "__main__":
    main()
