"""
Comprobación de memorización: ¿los modelos generan canales nuevos o copian los de entrenamiento?

Para cada canal generado se calcula la distancia al canal de entrenamiento más cercano (vecino más
próximo, NN) y se compara con la distancia típica entre dos canales reales independientes: cada canal de
train frente al resto de train excluyendo sus vecinos temporales (±--exclude medidas, el tiempo de
coherencia). Si un modelo memorizase, sus canales estarían mucho más cerca de train que esa referencia.
Los canales reales de test se muestran como segunda referencia.

Distancias (dos espacios):
  - "norm": distancia euclídea en el espacio normalizado por tap (TapScaler ajustado con train), dividida
    por sqrt(2L). Todos los taps pesan igual, también los de la cola.
  - "rel": distancia relativa en el espacio original, ||g - t|| / ||t|| (dominada por los taps fuertes).

Métricas por modelo (y para test como referencia):
  - mediana y percentil 5 de la distancia NN,
  - ratio = mediana(modelo) / mediana(referencia train-train): ~1 o mayor = no memoriza; << 1 = copia,
  - copias: % de canales con distancia NN menor que el percentil 1 de la referencia train-train.

Uso (desde la raíz del repositorio, después de evaluation.py):
    python evaluation_models/memorization.py
Resultados en <paths.save_dir>/memorization.txt, memorization.csv y fig_memorization_nn.{pdf,png}.
"""
import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
import evaluation as ev  # noqa: E402
from train_models.normalization import TapScaler  # noqa: E402


def nn_distance(queries, refs, batch=1000, q_pos=None, r_pos=None, exclude=0):
    """
    Distancia euclídea de cada fila de `queries` a la fila más cercana de `refs` (N, D).
    Si se dan las posiciones temporales (q_pos, r_pos), se excluyen las referencias a `exclude` o menos
    medidas de distancia (para comparar train consigo mismo sin los canales vecinos, casi idénticos).
    """
    r = torch.as_tensor(refs, dtype=torch.float64)
    r2 = (r ** 2).sum(1)
    out = []
    for i in range(0, len(queries), batch):
        q = torch.as_tensor(queries[i:i + batch], dtype=torch.float64)
        d2 = (q ** 2).sum(1, keepdim=True) + r2[None] - 2 * q @ r.T
        if q_pos is not None:
            near = np.abs(q_pos[i:i + batch][:, None] - r_pos[None]) <= exclude
            d2[torch.as_tensor(near)] = float("inf")
        dmin, idx = d2.clamp_min(0).min(dim=1)
        out.append((dmin.sqrt().numpy(), idx.numpy()))
    return np.concatenate([o[0] for o in out]), np.concatenate([o[1] for o in out])


def distances(x, train, scaler, **kw):
    """Distancias NN de x (N, 2, L) a train en los dos espacios (kw: exclusión de vecinos temporales)."""
    n, _, L = x.shape
    zn = scaler.transform(x).reshape(n, -1)
    zt = scaler.transform(train).reshape(len(train), -1)
    d_norm, _ = nn_distance(zn, zt, **kw)
    d_raw, idx = nn_distance(x.reshape(n, -1), train.reshape(len(train), -1), **kw)
    rel = d_raw / np.linalg.norm(train.reshape(len(train), -1)[idx], axis=1)
    return {"norm": d_norm / np.sqrt(2 * L), "rel": rel}


def main():
    parser = argparse.ArgumentParser(description="Comprobación de memorización (vecino más próximo)")
    parser.add_argument("--config", default=os.path.join(THIS_DIR, "evaluation_config.json"))
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--exclude", type=int, default=200,
                        help="Vecinos temporales excluidos en la referencia train-train (tiempo de coherencia)")
    parser.add_argument("--n-ref", type=int, default=2000, help="Canales de train usados como consulta")
    args = parser.parse_args()

    config = ev.load_config(args.config)
    data = np.load(ev.repo_path(config["data"]["path"])).astype(np.float32)
    split = np.load(ev.repo_path(config["data"]["split_file"]))
    train, test = data[split["train"]], data[split["test"]]
    scaler = TapScaler.fit(train)
    save_dir = ev.repo_path(config["paths"]["save_dir"])
    gen_dir = os.path.join(save_dir, "generated")
    models = args.models or config["evaluation"]["models"]

    print(f"Train {len(train)} | test {len(test)}")
    # Referencia principal: canales de train frente al resto de train sin sus vecinos temporales
    # (|i - j| <= --exclude). Es la distancia típica entre dos canales reales independientes de la misma
    # distribución. Test sirve de segunda referencia (está más lejos en el tiempo y el canal deriva).
    tr_idx = split["train"]
    q = np.random.default_rng(0).choice(len(train), min(args.n_ref, len(train)), replace=False)
    res = {"train": distances(train[q], train, scaler, q_pos=tr_idx[q], r_pos=tr_idx, exclude=args.exclude),
           "test": distances(test, train, scaler)}
    for m in models:
        path = os.path.join(gen_dir, f"h_generated_{m}.npy")
        if not os.path.exists(path):
            print(f"[{m}] no hay canales generados, se omite")
            continue
        res[m] = distances(np.load(path).astype(np.float32), train, scaler)
        print(f"[{m}] hecho")

    rows = []
    for name, d in res.items():
        row = {"model": {"train": "Real train", "test": "Real test"}.get(name) or ev.MODEL_LABELS[name]}
        for sp in ("norm", "rel"):
            ref = res["train"][sp]
            row[f"{sp}_median"] = float(np.median(d[sp]))
            row[f"{sp}_p5"] = float(np.percentile(d[sp], 5))
            row[f"{sp}_ratio"] = float(np.median(d[sp]) / np.median(ref))
            row[f"{sp}_copies_pct"] = float(np.mean(d[sp] < np.percentile(ref, 1)) * 100)
        rows.append(row)

    with open(os.path.join(save_dir, "memorization.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cols = [("model", "Modelo", "s")] + [(f"{sp}_{k}", f"{sp} {lbl}", fmt) for sp in ("norm", "rel")
                                         for k, lbl, fmt in (("median", "mediana", ".4f"), ("p5", "p5", ".4f"),
                                                             ("ratio", "ratio", ".2f"),
                                                             ("copies_pct", "copias %", ".2f"))]
    lines = ["Distancia al canal de entrenamiento más cercano (vecino más próximo)",
             "norm: espacio normalizado por tap, / sqrt(2L) | rel: ||g - t|| / ||t|| en el espacio original",
             f"Real train: canales de train frente al resto de train sin sus vecinos (±{args.exclude} medidas)",
             "ratio = mediana / mediana de Real train (~1 o mayor: no memoriza; << 1: copia train)",
             "copias % = canales con distancia menor que el percentil 1 de Real train (1% por definición)", "",
             "".join(f"{lbl:>16}" for _, lbl, _ in cols)]
    for r in rows:
        lines.append("".join(f"{r[k]:>16{fmt}}" for k, _, fmt in cols))
    txt = "\n".join(lines)
    with open(os.path.join(save_dir, "memorization.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n" + txt)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, sp, xl in zip(axes, ("norm", "rel"), ("NN distance (per-tap normalized)",
                                                  "Relative NN distance ||g - t|| / ||t||")):
        for name, d in res.items():
            v = np.sort(d[sp])
            if name == "train":
                ax.plot(v, np.arange(1, len(v) + 1) / len(v), color="black", lw=2,
                        label=f"Real train (excl. ±{args.exclude})")
            elif name == "test":
                ax.plot(v, np.arange(1, len(v) + 1) / len(v), color="black", lw=1.5, ls=":", label="Real test")
            else:
                ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=ev.COLORS[name],
                        ls=ev.LINESTYLES.get(name, "--"), lw=1.3, label=ev.MODEL_LABELS[name])
        ev._style(ax, xl, "CDF", legend=ax is axes[0], loc="lower right")
    ev._save(fig, save_dir, "fig_memorization_nn")
    print(f"\nResultados en {save_dir}/memorization.txt")


if __name__ == "__main__":
    main()
