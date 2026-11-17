"""
Evaluación de utilidad (sic_cnn.py): combinación de las K particiones de la validación cruzada por tramos.

Para cada partición k, sic_results.npz guarda la cancelación por canal de test (real) de cada estimador:
ls, avg, wiener, lmmse_<fuente> y cnn_<fuente>. La cifra de la partición es la media por canal y la de la
validación cruzada es la **media sobre las K particiones**.

Bootstrap estratificado por tramos (como cv_stats.py): en cada réplica, dentro de cada partición, los tramos
de test se remuestrean enteros con reemplazo (tantos como hay); se recalcula la media por canal de cada
partición y se promedian las particiones. Las diferencias frente a una referencia (cnn_gauss, cnn_real) son
emparejadas: las mismas réplicas para ambos estimadores. IC95 = percentiles 2.5–97.5.

Ojo: el bootstrap solo cubre la variabilidad del test. No cubre la del entrenamiento (una sola semilla por
fuente y partición) ni la del generador; la dispersión entre particiones da una idea de esta última.

Uso (desde la raíz):
    python evaluation_models/sic_stats.py
Salida en eval_results/sodupe_180_segcv/: sic_cv.txt y sic_cv.csv.
"""
import argparse
import csv
import glob
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFS = ("cnn_gauss", "cnn_real")


def main():
    p = argparse.ArgumentParser(description="SIC CNN: validación cruzada por tramos")
    p.add_argument("--dir", default="eval_results/sodupe_180_segcv")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="carpeta de salida (por defecto, --dir)")
    args = p.parse_args()

    base = os.path.join(ROOT, args.dir)
    files = sorted(glob.glob(os.path.join(base, "fold*", "sic", "sic_results.npz")))
    folds = [np.load(f) for f in files]
    keys = [k for k in folds[0].files if k not in ("channel", "segment", "psi")]
    keys = [k for k in keys if all(k in d.files for d in folds)]

    # Por partición: sumas y cuentas por tramo -> el remuestreo de tramos da la media por canal.
    sums, counts = [], []
    for d in folds:
        segs, inv = np.unique(d["segment"], return_inverse=True)
        counts.append(np.bincount(inv, minlength=len(segs)).astype(float))
        sums.append({k: np.bincount(inv, weights=d[k], minlength=len(segs)) for k in keys})

    def cv_mean(w):  # w: pesos de remuestreo por tramo, uno por partición
        return {k: np.mean([(w[i] @ sums[i][k]) / (w[i] @ counts[i]) for i in range(len(folds))])
                for k in keys}

    fold_vals = {k: [s[k].sum() / c.sum() for s, c in zip(sums, counts)] for k in keys}
    point = cv_mean([np.ones_like(c) for c in counts])

    rng = np.random.default_rng(args.seed)
    boot = {k: np.empty(args.n_boot) for k in keys}
    for b in range(args.n_boot):
        w = [np.bincount(rng.integers(0, len(c), len(c)), minlength=len(c)).astype(float) for c in counts]
        for k, v in cv_mean(w).items():
            boot[k][b] = v

    def ci(x):
        return np.percentile(x, [2.5, 97.5])

    order = sorted(keys, key=lambda k: -point[k])
    lines = [f"Validación cruzada por tramos, {len(folds)} particiones, "
             f"{int(sum(c.size for c in counts))} tramos de test, {int(sum(c.sum() for c in counts))} canales; "
             f"bootstrap estratificado por tramos, {args.n_boot} réplicas",
             "Cancelación media (dB) sobre particiones [IC95]; valor por partición; diferencias emparejadas.",
             "Solo variabilidad del test: no incluye la del entrenamiento (una semilla por fuente y partición).", ""]
    hdr = f"{'estimador':16s} {'CV':>6s} {'IC95':>15s}  " + " ".join(f"{'p' + str(i):>6s}" for i in range(len(folds)))
    hdr += "".join(f"  {'Δ ' + r:>16s} {'IC95':>15s} {'P(>)':>5s} {'gana':>5s}" for r in REFS)
    lines.append(hdr)
    rows = []
    for k in order:
        lo, hi = ci(boot[k])
        s = f"{k:16s} {point[k]:6.2f} [{lo:6.2f},{hi:6.2f}]  " + " ".join(f"{v:6.2f}" for v in fold_vals[k])
        row = {"estimador": k, "cv": point[k], "ci_lo": lo, "ci_hi": hi,
               **{f"fold{i}": v for i, v in enumerate(fold_vals[k])}}
        for r in REFS:
            if r not in keys:
                s += " " * 46
                continue
            dd = boot[k] - boot[r]
            dl, dh = ci(dd)
            delta = point[k] - point[r]
            wins = sum(a > b for a, b in zip(fold_vals[k], fold_vals[r]))
            s += f"  {delta:+16.2f} [{dl:+6.2f},{dh:+6.2f}] {np.mean(dd > 0):5.2f} {wins:3d}/{len(folds)}"
            row.update({f"d_{r}": delta, f"d_{r}_lo": dl, f"d_{r}_hi": dh, f"p_gt_{r}": np.mean(dd > 0),
                        f"folds_gt_{r}": wins})
        lines.append(s)
        rows.append(row)

    txt = "\n".join(lines)
    print(txt)
    out = args.out or base
    with open(os.path.join(out, "sic_cv.txt"), "w") as f:
        f.write(txt + "\n")
    with open(os.path.join(out, "sic_cv.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(dict.fromkeys(c for r in rows for c in r)))
        wr.writeheader()
        wr.writerows(rows)


if __name__ == "__main__":
    main()
