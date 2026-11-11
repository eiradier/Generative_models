"""
Comprobación de memorización: ¿los modelos generan canales nuevos o copian los de entrenamiento?

Para cada canal generado se calcula la distancia al canal de entrenamiento más cercano (vecino más
próximo, NN) y se compara con la misma distancia para los canales reales de test, que el modelo no ha
visto. Si un modelo memorizase, sus canales estarían mucho más cerca de train que los de test.

Distancias (dos espacios):
  - "norm": distancia euclídea en el espacio normalizado por tap (TapScaler ajustado con train), dividida
    por sqrt(2L). Todos los taps pesan igual, también los de la cola.
  - "rel": distancia relativa en el espacio original, ||g - t|| / ||t|| (dominada por los taps fuertes).

Métricas por modelo (y para test como referencia):
  - mediana y percentil 5 de la distancia NN,
  - ratio = mediana(modelo) / mediana(test): ~1 o mayor = no memoriza; << 1 = copia train,
  - copias: % de canales generados con distancia NN menor que el percentil 1 de test.

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


def nn_distance(queries, refs, batch=1000):
    """Distancia euclídea de cada fila de `queries` a la fila más cercana de `refs` (N, D)."""
    r = torch.as_tensor(refs, dtype=torch.float64)
    r2 = (r ** 2).sum(1)
    out = []
    for i in range(0, len(queries), batch):
        q = torch.as_tensor(queries[i:i + batch], dtype=torch.float64)
        d2 = (q ** 2).sum(1, keepdim=True) + r2[None] - 2 * q @ r.T
        dmin, idx = d2.clamp_min(0).min(dim=1)
        out.append((dmin.sqrt().numpy(), idx.numpy()))
    return np.concatenate([o[0] for o in out]), np.concatenate([o[1] for o in out])


def distances(x, train, scaler):
    """Distancias NN de x (N, 2, L) a train en los dos espacios."""
    n, _, L = x.shape
    zn = scaler.transform(x).reshape(n, -1)
    zt = scaler.transform(train).reshape(len(train), -1)
    d_norm, _ = nn_distance(zn, zt)
    d_raw, idx = nn_distance(x.reshape(n, -1), train.reshape(len(train), -1))
    rel = d_raw / np.linalg.norm(train.reshape(len(train), -1)[idx], axis=1)
    return {"norm": d_norm / np.sqrt(2 * L), "rel": rel}


def main():
    parser = argparse.ArgumentParser(description="Comprobación de memorización (vecino más próximo)")
    parser.add_argument("--config", default=os.path.join(THIS_DIR, "evaluation_config.json"))
    parser.add_argument("--models", nargs="+", default=None)
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
    res = {"test": distances(test, train, scaler)}
    for m in models:
        path = os.path.join(gen_dir, f"h_generated_{m}.npy")
        if not os.path.exists(path):
            print(f"[{m}] no hay canales generados, se omite")
            continue
        res[m] = distances(np.load(path).astype(np.float32), train, scaler)
        print(f"[{m}] hecho")

    rows = []
    for name, d in res.items():
        row = {"model": "Real test" if name == "test" else ev.MODEL_LABELS[name]}
        for sp in ("norm", "rel"):
            ref = res["test"][sp]
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
             "ratio = mediana / mediana de test (~1 o mayor: no memoriza; << 1: copia train)",
             "copias % = canales con distancia menor que el percentil 1 de test (test: 1% por definición)", "",
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
            if name == "test":
                ax.plot(v, np.arange(1, len(v) + 1) / len(v), color="black", lw=2, label="Real (test)")
            else:
                ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=ev.COLORS[name],
                        ls=ev.LINESTYLES.get(name, "--"), lw=1.3, label=ev.MODEL_LABELS[name])
        ev._style(ax, xl, "CDF", legend=ax is axes[0], loc="lower right")
    ev._save(fig, save_dir, "fig_memorization_nn")
    print(f"\nResultados en {save_dir}/memorization.txt")


if __name__ == "__main__":
    main()
