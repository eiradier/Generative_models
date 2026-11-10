"""
Barrido de los parámetros de muestreo del DDIM (pasos y eta) con un modelo ya entrenado.
No hace falta reentrenar: cada combinación solo genera canales y los compara con la referencia.

Uso (desde la raíz del repositorio):
    python evaluation_models/sweep_sampling.py
    python evaluation_models/sweep_sampling.py --steps 25 50 100 250 --eta 0 0.5 1 --n 1000 --subset val

Resultados en <paths.save_dir>/sweep_sampling_ddim.csv y .txt (ordenados por selection_score).
Con el mejor resultado, actualiza models.ddim.sampling_steps y models.ddim.eta en
evaluation_config.json.
"""
import argparse
import csv
import itertools
import os
import sys
import time

import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
import evaluation as ev  # noqa: E402


def load_reference(config, subset):
    data = np.load(ev.repo_path(config["data"]["path"])).astype(np.float32)
    if subset != "all":
        data = data[np.load(ev.repo_path(config["data"]["split_file"]))[subset]]
    return data


def main():
    parser = argparse.ArgumentParser(description="Barrido de pasos/eta del muestreo DDIM")
    parser.add_argument("--config", default=os.path.join(THIS_DIR, "evaluation_config.json"))
    parser.add_argument("--checkpoint", default=None, help="Por defecto models.ddim.checkpoint")
    parser.add_argument("--steps", type=int, nargs="+", default=[25, 50, 100, 250])
    parser.add_argument("--eta", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    parser.add_argument("--n", type=int, default=1000, help="Canales generados por combinación")
    parser.add_argument("--subset", default=None, help="Referencia: val (por defecto si hay split), test o all")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = ev.load_config(args.config)
    seed = config["experiment"]["seed"]
    subset = args.subset or ("val" if config["data"].get("split_file") else "all")
    ref = load_reference(config, subset)
    h_ref = ev.to_complex(ref)
    dt = ev.sampling_period(config)
    L = ref.shape[-1]

    mcfg = dict(config["models"]["ddim"])
    mcfg["checkpoint"] = ev.repo_path(args.checkpoint or mcfg["checkpoint"])
    print(f"Checkpoint: {mcfg['checkpoint']} | referencia: {subset} ({len(ref)} canales) | n={args.n}")

    rows = []
    for steps, eta in itertools.product(args.steps, args.eta):
        t0 = time.time()
        cfg = {**mcfg, "sampling_steps": steps, "eta": eta}
        ev.set_seed(seed)
        fake = ev.generate_signals("ddim", cfg, args.n, L, args.device, ref, seed)
        m = ev.quick_metrics(h_ref, ev.to_complex(fake), dt, seed=seed)
        row = {"steps": steps, "eta": eta, "time_s": round(time.time() - t0, 1),
               **{k: m[k] for k in ev.SUMMARY_KEYS}}
        rows.append(row)
        print(f"steps={steps:4d} eta={eta:.2f} | score {m['selection_score']:.3f} | "
              f"PDP {m['pdp_mae_db']:.2f} dB | KS rms {m['rms_delay_ks']:.3f} | KS |h| {m['mag_ks']:.3f} | "
              f"{row['time_s']} s", flush=True)

    rows.sort(key=lambda r: r["selection_score"])
    save_dir = ev.repo_path(config["paths"]["save_dir"])
    os.makedirs(save_dir, exist_ok=True)
    base = os.path.join(save_dir, "sweep_sampling_ddim")
    with open(base + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(base + ".txt", "w", encoding="utf-8") as f:
        f.write(f"Barrido de muestreo DDIM ({mcfg['checkpoint']}), referencia '{subset}', n={args.n}\n")
        f.write("Ordenado por selection_score (menor es mejor)\n\n")
        cols = list(rows[0])
        f.write("".join(f"{c:>18}" for c in cols) + "\n")
        for r in rows:
            f.write("".join(f"{r[c]:>18.4g}" if isinstance(r[c], float) else f"{r[c]:>18}" for c in cols) + "\n")
    best = rows[0]
    print(f"\nMejor: steps={best['steps']} eta={best['eta']} (score {best['selection_score']:.3f}). "
          f"Resultados en {base}.csv/.txt")


if __name__ == "__main__":
    main()
