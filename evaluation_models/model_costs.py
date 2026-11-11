"""
Coste computacional de cada modelo: número de parámetros, tiempo de generación y tiempo de entrenamiento.

- Parámetros: se cuentan en el checkpoint (generador en las GAN, U-Net en los de difusión).
- Generación: tiempo medido generando --n canales (tras un calentamiento), en el dispositivo indicado.
  Los modelos clásicos (baselines.py) incluyen el ajuste con train.
- Entrenamiento: se lee de <paths.save_dir>/training_log.json (épocas, mejor época y minutos de cada
  modelo, tomados de los logs del entrenamiento).

Uso (desde la raíz del repositorio):
    python evaluation_models/model_costs.py
    python evaluation_models/model_costs.py --n 500 --device cpu
Resultados en <paths.save_dir>/model_costs.txt y model_costs.csv.
"""
import argparse
import csv
import json
import os
import platform
import sys
import time

import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
import evaluation as ev  # noqa: E402
from baselines import BASELINES, generate_baseline  # noqa: E402


def count_params(model_type, checkpoint):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = ckpt["G"] if model_type in ("gan", "dcgan", "wgan") else ckpt
    return int(sum(v.numel() for k, v in state.items()
                   if torch.is_tensor(v) and "running" not in k and "num_batches" not in k))


def cpu_name():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "CPU"


def main():
    parser = argparse.ArgumentParser(description="Parámetros, tiempo de generación y de entrenamiento")
    parser.add_argument("--config", default=os.path.join(THIS_DIR, "evaluation_config.json"))
    parser.add_argument("--n", type=int, default=500, help="Canales generados para medir el tiempo")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = ev.load_config(args.config)
    seed = config["experiment"]["seed"]
    data = np.load(ev.repo_path(config["data"]["path"])).astype(np.float32)
    split = np.load(ev.repo_path(config["data"]["split_file"]))
    train = data[split["train"]]
    L = data.shape[-1]
    save_dir = ev.repo_path(config["paths"]["save_dir"])
    log_path = os.path.join(save_dir, "training_log.json")
    train_log = json.load(open(log_path)) if os.path.exists(log_path) else {}
    device_name = torch.cuda.get_device_name(0) if args.device.startswith("cuda") else \
        f"{cpu_name()} ({torch.get_num_threads()} hilos)"

    rows = []
    for m in config["evaluation"]["models"]:
        row = {"model": ev.MODEL_LABELS[m]}
        if m in BASELINES:
            row["params"] = None
            t0 = time.perf_counter()
            generate_baseline(m, train, args.n, seed)
            gen_s = time.perf_counter() - t0
        else:
            mcfg = {**config["models"][m], "checkpoint": ev.repo_path(config["models"][m]["checkpoint"])}
            row["params"] = count_params(m, mcfg["checkpoint"])
            ev.generate_signals(m, mcfg, 8, L, args.device, train, seed, batch_size=8)  # calentamiento
            t0 = time.perf_counter()
            ev.generate_signals(m, mcfg, args.n, L, args.device, train, seed, batch_size=500)
            gen_s = time.perf_counter() - t0
        row["gen_ms_per_channel"] = gen_s / args.n * 1e3
        row["gen_s_1000"] = gen_s / args.n * 1000
        tl = train_log.get(m, {})
        row["epochs"] = tl.get("epochs")
        row["best_epoch"] = tl.get("best_epoch")
        row["train_min"] = tl.get("train_min")
        rows.append(row)
        print(f"[{row['model']}] {row['gen_ms_per_channel']:.2f} ms/canal", flush=True)

    with open(os.path.join(save_dir, "model_costs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    fmt = lambda v, f: "-" if v is None else format(v, f)  # noqa: E731
    lines = [f"Coste computacional ({device_name}, PyTorch {torch.__version__})",
             f"Generación medida con {args.n} canales; entrenamiento: {os.path.basename(log_path)}", "",
             f"{'Modelo':<12}{'Parámetros':>12}{'ms/canal':>12}{'s/1000 can.':>13}{'Épocas':>9}"
             f"{'Mejor ép.':>11}{'Entren. (min)':>15}"]
    for r in rows:
        lines.append(f"{r['model']:<12}{fmt(r['params'], ','):>12}{r['gen_ms_per_channel']:>12.2f}"
                     f"{r['gen_s_1000']:>13.1f}{fmt(r['epochs'], 'd'):>9}{fmt(r['best_epoch'], 'd'):>11}"
                     f"{fmt(r['train_min'], '.1f'):>15}")
    notes = train_log.get("_notes")
    if notes:
        lines += ["", *notes]
    txt = "\n".join(lines)
    with open(os.path.join(save_dir, "model_costs.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n" + txt)


if __name__ == "__main__":
    main()
