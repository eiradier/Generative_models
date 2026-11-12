"""
Resultados de la validación cruzada por tramos (make_segment_folds.py): combina las K particiones.

Para cada partición k se calculan las métricas de seed_stats.py (PDP MAE, KS de |h| y de los rasgos por
canal, |Δlog10 BER|) de cada fuente frente al test de esa partición: modelos generativos y clásicos
entrenados/ajustados con su train, y referencias reales train y val de la misma partición. La cifra de
validación cruzada es la **media sobre las K particiones**.

Bootstrap estratificado por tramos: en cada réplica, dentro de cada partición, test, train y val se
remuestrean por tramos enteros (con reemplazo; tantos tramos como hay) y los generados canal a canal;
se recalculan las métricas por partición y se promedian. Así el test efectivo son todos los tramos
(cada uno es test en una partición). IC95 = percentiles 2.5–97.5; P(< Gaussian) emparejada.

Uso (desde la raíz, con eval_results/sodupe_180_segcv/fold<k>/ evaluadas):
    python evaluation_models/cv_stats.py --cache <carpeta>
Salida en eval_results/sodupe_180_segcv/: cv_folds.txt/.csv (métricas por partición) y cv_bootstrap.txt/.csv.
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

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
import evaluation as ev  # noqa: E402
from seed_stats import CLASSIC, GEN_MODELS, REF, Source, block_resample, metric_values  # noqa: E402

_REPLICATE = None


def _run_replicate(b):
    return b, _REPLICATE(b)


def main():
    p = argparse.ArgumentParser(description="Validación cruzada por tramos: combinación de particiones")
    p.add_argument("--dir", default="eval_results/sodupe_180_segcv")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    p.add_argument("--cache", default=None)
    args = p.parse_args()

    base = ev.repo_path(args.dir)
    folds = sorted(d for d in glob.glob(os.path.join(base, "fold*"))
                   if os.path.exists(os.path.join(d, "metrics.pkl")))
    print(f"{len(folds)} particiones evaluadas:", *folds, sep="\n  ")
    cache = ev.repo_path(args.cache) if args.cache else None
    if cache:
        os.makedirs(cache, exist_ok=True)

    fold_data = []
    for fd in folds:
        cfg = ev.load_config(os.path.join(fd, "evaluation_config.json"))
        data = np.load(ev.repo_path(cfg["data"]["path"])).astype(np.float32)
        split = np.load(ev.repo_path(cfg["data"]["split_file"]))
        seg = split["segment"]
        dt = ev.sampling_period(cfg)
        seed = cfg["experiment"]["seed"]
        ber_cfg = {"snr_db": cfg["ber"]["snr_db"], "modulations": cfg["ber"]["modulations"]}
        mag_edges = np.concatenate([[-1e9], np.linspace(-120, 20, 1401), [1e9]])
        name = os.path.basename(fd)

        def get(key, loader):
            path = os.path.join(cache, f"cv_{name}_{key}.pkl") if cache else None
            if path and os.path.exists(path):
                with open(path, "rb") as f:
                    return pickle.load(f)
            src = Source(loader(), dt, cfg, mag_edges, ber_cfg, seed, 10 ** 9)
            if path:
                with open(path, "wb") as f:
                    pickle.dump(src, f)
            print(f"  {name} {key}", flush=True)
            return src

        srcs, labels = {}, {}
        for k in ("test", "train", "val"):
            idx = np.sort(split[k])
            srcs[f"real_{k}"] = get(f"real_{k}", lambda idx=idx: ev.to_complex(data[idx]))
            labels[k] = seg[idx]
        for m in GEN_MODELS + CLASSIC:
            g = os.path.join(fd, "generated", f"h_generated_{m}.npy")
            if os.path.exists(g):
                srcs[m] = get(m, lambda g=g: ev.to_complex(np.load(g)))
        test = srcs["real_test"]
        mask = test.p.mean(0) > test.p.mean(0).max() * 1e-6
        fold_data.append({"name": name, "srcs": srcs, "labels": labels, "mask": mask,
                          "feat_keys": [k for k in test.feat if not k.startswith("_")]})

    models = [m for m in GEN_MODELS + CLASSIC if all(m in f["srcs"] for f in fold_data)]
    names = ["real_train", "real_val"] + models
    seed = 42

    def fold_metrics(f, rng):
        test = f["srcs"]["real_test"]
        ridx = block_resample(test.n, None, rng, f["labels"]["test"]) if rng is not None else np.arange(test.n)
        res = {}
        for n in names:
            s = f["srcs"][n]
            if rng is None:
                idx = np.arange(s.n)
            elif n.startswith("real_"):
                idx = block_resample(s.n, None, rng, f["labels"][n[5:]])
            else:
                idx = rng.integers(0, s.n, s.n)
            res[n] = metric_values(s, idx, test, ridx, f["mask"], f["feat_keys"])
        return res

    def cv_metrics(rng):
        per = [fold_metrics(f, rng) for f in fold_data]
        return {n: {k: float(np.mean([r[n][k] for r in per])) for k in per[0][n]} for n in names}, per

    point, per_fold = cv_metrics(None)
    keys = list(point["real_train"])

    # Métricas por partición (datos completos)
    rows = [{"fold": f["name"], "source": n, **per_fold[i][n]} for i, f in enumerate(fold_data) for n in names]
    with open(os.path.join(base, "cv_folds.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    label = {"real_train": "Real train", "real_val": "Real val", **ev.MODEL_LABELS}
    lines = ["Métricas por partición (frente al test de cada partición) y media de validación cruzada", ""]
    for k in keys:
        lines.append(f"== {k}")
        lines.append(f"  {'':<12}" + "".join(f"{f['name']:>9}" for f in fold_data) + f"{'media':>9}{'std':>8}")
        for n in names:
            v = [per_fold[i][n][k] for i in range(len(fold_data))]
            lines.append(f"  {label[n]:<12}" + "".join(f"{x:>9.4f}" for x in v)
                         + f"{np.mean(v):>9.4f}{np.std(v, ddof=1) if len(v) > 1 else 0:>8.4f}")
    with open(os.path.join(base, "cv_folds.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    # Bootstrap estratificado por tramos
    global _REPLICATE
    _REPLICATE = lambda b: cv_metrics(np.random.default_rng([seed, b]))[0]  # noqa: E731
    done = {}
    part = os.path.join(cache, "cv_replicates.pkl") if cache else None
    if part and os.path.exists(part):
        with open(part, "rb") as fh:
            done = pickle.load(fh)
        if len(next(iter(done.values()), {})) and set(next(iter(done.values()))) != set(names):
            done = {}
    todo = [b for b in range(args.n_boot) if b not in done]
    print(f"Réplicas: {len(done)} hechas, {len(todo)} pendientes", flush=True)
    with mp.get_context("fork").Pool(args.workers, initializer=torch.set_num_threads, initargs=(1,)) as pool:
        for i, (b, res) in enumerate(pool.imap_unordered(_run_replicate, todo, chunksize=4)):
            done[b] = res
            if (i + 1) % 50 == 0 or i + 1 == len(todo):
                print(f"  réplica {len(done)}/{args.n_boot}", flush=True)
                if part:
                    with open(part + ".tmp", "wb") as fh:
                        pickle.dump(done, fh)
                    os.replace(part + ".tmp", part)
    boot = {n: {k: np.array([done[b][n][k] for b in range(args.n_boot)]) for k in keys} for n in names}

    rows = []
    for k in keys:
        for n in names:
            rows.append({"metric": k, "source": n, "mean": point[n][k],
                         "ci_low": float(np.percentile(boot[n][k], 2.5)),
                         "ci_high": float(np.percentile(boot[n][k], 97.5)),
                         "p_better_than_gauss": float(np.mean(boot[n][k] < boot[REF][k])) if n != REF else np.nan})
    with open(os.path.join(base, "cv_bootstrap.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n_seg = sum(len(np.unique(f["labels"]["test"])) for f in fold_data)
    lines = [f"Validación cruzada por tramos: {len(fold_data)} particiones, {n_seg} tramos de test en total. "
             f"Media sobre particiones; bootstrap estratificado por tramos ({args.n_boot} réplicas).",
             "P(<Gauss): fracción de réplicas en que la fuente mejora al gaussiano (emparejada); ** > 0.975, "
             "†† < 0.025. Menor es mejor.", ""]
    for k in keys:
        lines.append(f"== {k}")
        for r in [x for x in rows if x["metric"] == k]:
            pb = r["p_better_than_gauss"]
            mark = "" if np.isnan(pb) else ("**" if pb > 0.975 else "††" if pb < 0.025 else "")
            ps = "" if np.isnan(pb) else f"{pb:.3f}{mark}"
            lines.append(f"  {label[r['source']]:<12} {r['mean']:>9.4f}  IC95 [{r['ci_low']:.4f}, "
                         f"{r['ci_high']:.4f}]   P(<Gauss) {ps}")
    with open(os.path.join(base, "cv_bootstrap.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Resultados en {base}/cv_folds.txt y cv_bootstrap.txt")


if __name__ == "__main__":
    main()
