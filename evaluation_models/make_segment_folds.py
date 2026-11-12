"""
Validación cruzada por tramos continuos: reparte los tramos de recordings.py (segment_filtered) en K
particiones. En la partición k, test = los tramos del grupo k; val = tramos enteros elegidos al azar entre
los demás hasta ~val_frac del dataset; train = el resto. Ningún tramo se reparte entre conjuntos, así que
no hacen falta huecos (los cortes entre tramos son discontinuidades del canal, §2.3 de PAPER_NOTES).

Los grupos se equilibran en número de canales (tramos ordenados por tamaño, cada uno al grupo con menos
canales; desempates al azar con la semilla).

Uso (desde la raíz del repositorio, tras recordings.py):
    python evaluation_models/make_segment_folds.py
Salida: data/datos_loopback/ch_sodupe_180_time_L200_fold<k>.npz con train / val / test (índices del
dataset filtrado, como el split por bloques) y segment (tramo de cada canal), y un resumen
<out_dir>/folds.txt.
"""
import argparse
import os

import numpy as np


def main():
    p = argparse.ArgumentParser(description="Particiones de validación cruzada por tramos")
    p.add_argument("--recordings", default="eval_results/sodupe_180_split_g200/recordings.npz")
    p.add_argument("--out-prefix", default="data/datos_loopback/ch_sodupe_180_time_L200_fold")
    p.add_argument("--out-dir", default="eval_results/sodupe_180_segcv")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    seg = np.load(args.recordings)["segment_filtered"]
    ids, sizes = np.unique(seg, return_counts=True)
    rng = np.random.default_rng(args.seed)

    # Asignación equilibrada de tramos a grupos (mayor primero, al grupo con menos canales)
    order = np.lexsort((rng.random(len(ids)), -sizes))
    group = np.empty(len(ids), dtype=int)
    load = np.zeros(args.folds, dtype=int)
    for i in order:
        g = int(np.argmin(load + rng.random(args.folds) * 1e-3))
        group[i] = g
        load[g] += sizes[i]

    os.makedirs(args.out_dir, exist_ok=True)
    lines = [f"Validación cruzada por tramos: {len(ids)} tramos, {len(seg)} canales, {args.folds} particiones, "
             f"val ~{args.val_frac:.0%} (tramos enteros), semilla {args.seed}", ""]
    target_val = args.val_frac * len(seg)
    for k in range(args.folds):
        test_ids = ids[group == k]
        rest = ids[group != k]
        val_ids, n = [], 0
        for s in rng.permutation(rest):
            if n >= target_val:
                break
            val_ids.append(s)
            n += sizes[ids == s][0]
        val_ids = np.array(sorted(val_ids))
        train_ids = np.setdiff1d(rest, val_ids)
        split = {name: np.where(np.isin(seg, v))[0] for name, v in
                 [("train", train_ids), ("val", val_ids), ("test", test_ids)]}
        assert sum(len(v) for v in split.values()) == len(seg)
        path = f"{args.out_prefix}{k}.npz"
        np.savez(path, **split, segment=seg)
        lines.append(f"Partición {k} ({path}): " + ", ".join(
            f"{name} {len(split[name])} canales / {len(v)} tramos"
            for name, v in [("train", train_ids), ("val", val_ids), ("test", test_ids)]))
    lines.append("")
    lines.append("Tramos de test por partición: " + "; ".join(
        f"{k}: {sorted(ids[group == k].tolist())}" for k in range(args.folds)))
    txt = "\n".join(lines)
    with open(os.path.join(args.out_dir, "folds.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
