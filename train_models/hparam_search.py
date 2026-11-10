"""
Búsqueda de hiperparámetros: entrena una configuración por prueba, genera canales y los compara con
el conjunto de validación. Los resultados se ordenan por `selection_score` (ver
evaluation.quick_metrics; menor es mejor).

Uso (desde la raíz del repositorio):
    python train_models/hparam_search.py train_models/hparam_specs/ddim.json
    python train_models/hparam_search.py train_models/hparam_specs/ddim.json --dry-run   # lista las pruebas

Es reanudable: si se interrumpe, al relanzarlo se saltan las pruebas terminadas y la que estaba a
medias continúa desde su última época (DDPM/DDIM/WGAN guardan last.pth).

Formato del fichero de especificación (JSON):
{
  "model": "ddim",                                   # gan | dcgan | wgan | ddpm | ddim
  "base_config": "train_models/models_configs/dm_config.json",
  "out_dir": "hpsearch/ddim",
  "overrides": {"training.epochs": 150},             # se aplican a todas las pruebas
  "search": "grid",                                  # grid | random
  "n_trials": 8,                                     # solo para random
  "params": {"training.lr": [1e-4, 2e-4, 5e-4],     # claves con puntos = rutas en la config
             "diffusion.lambda_psd": [0, 0.05]},
  "eval": {"n_samples": 1000, "subset": "val", "sampling_steps": 100, "eta": 0.0}
}
"""
import argparse
import copy
import csv
import itertools
import json
import os
import random
import subprocess
import sys
import time

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "evaluation_models"))
import evaluation as ev  # noqa: E402

CHECKPOINT = {
    "gan": "GAN_Conv1D/best_model.pth",
    "dcgan": "DCGAN_Conv1D/model_best.pth",
    "wgan": "WGAN_Conv1D/best_model.pth",
    "ddpm": "DDPM/best_model.pth",
    "ddim": "DDIM/best_model.pth",
}


def set_path(cfg, dotted, value):
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def get_path(cfg, dotted, default=None):
    d = cfg
    for k in dotted.split("."):
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def make_trials(spec):
    names = list(spec["params"])
    combos = [dict(zip(names, vals)) for vals in itertools.product(*(spec["params"][n] for n in names))]
    if spec.get("search", "grid") == "random":
        rng = random.Random(spec.get("seed", 0))
        combos = rng.sample(combos, min(spec.get("n_trials", 10), len(combos)))
    return combos


def generation_cfg(model, trial_cfg, ckpt, ev_spec):
    cfg = {"checkpoint": ckpt}
    if model in ("ddpm", "ddim"):
        cfg["T"] = get_path(trial_cfg, "diffusion.T", 1000)
        cfg["time_emb_dim"] = get_path(trial_cfg, "model.time_emb_dim", 64)
        cfg["sampling_steps"] = ev_spec.get("sampling_steps", 100)
        cfg["eta"] = ev_spec.get("eta", 0.0)
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Búsqueda de hiperparámetros")
    parser.add_argument("spec")
    parser.add_argument("--dry-run", action="store_true", help="Solo lista las pruebas")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    os.chdir(REPO_ROOT)

    spec = json.load(open(args.spec))
    model = spec["model"]
    base = json.load(open(spec["base_config"]))
    out_dir = spec["out_dir"]
    ev_spec = spec.get("eval", {})
    trials = make_trials(spec)

    print(f"{len(trials)} pruebas para {model.upper()} -> {out_dir}")
    for i, t in enumerate(trials):
        print(f"  trial_{i:03d}: {t}")
    if args.dry_run:
        return

    os.makedirs(out_dir, exist_ok=True)
    results_csv = os.path.join(out_dir, "results.csv")
    done = set()
    if os.path.exists(results_csv):
        done = {row["trial"] for row in csv.DictReader(open(results_csv))}

    # Referencia (validación) en escala original
    data_path = get_path(base, "dataset.path")
    split_file = spec.get("overrides", {}).get("dataset.split_file", get_path(base, "dataset.split_file"))
    subset = ev_spec.get("subset", "val")
    ref = np.load(data_path).astype(np.float32)
    if split_file:
        ref = ref[np.load(split_file)[subset]]
    else:
        print("AVISO: sin dataset.split_file se evalúa contra todo el dataset (incluido el de entrenamiento)")
    h_ref = ev.to_complex(ref)
    dt = 7.0 / (8.0 * ev_spec.get("bandwidth_hz", 8e6))
    device = args.device or ("cuda" if ev.torch.cuda.is_available() else "cpu")
    seed = get_path(base, "experiment.seed", 42)

    for i, params in enumerate(trials):
        name = f"trial_{i:03d}"
        if name in done:
            print(f"[{name}] ya evaluado, se salta")
            continue
        trial_dir = os.path.join(out_dir, name)
        os.makedirs(trial_dir, exist_ok=True)

        cfg = copy.deepcopy(base)
        for k, v in {**spec.get("overrides", {}), **params}.items():
            set_path(cfg, k, v)
        set_path(cfg, "paths.save_dir", trial_dir)
        cfg_path = os.path.join(trial_dir, "config.json")
        json.dump(cfg, open(cfg_path, "w"), indent=2)

        print(f"\n[{name}] {params}", flush=True)
        t0 = time.time()
        log = open(os.path.join(trial_dir, "train.log"), "a")
        r = subprocess.run([sys.executable, "-u", os.path.join(THIS_DIR, "train.py"), "--model", model,
                            "--config", cfg_path], stdout=log, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            print(f"[{name}] ERROR en el entrenamiento (ver {trial_dir}/train.log)")
            continue
        train_min = (time.time() - t0) / 60

        ckpt = os.path.join(trial_dir, CHECKPOINT[model])
        ev.set_seed(seed)
        fake = ev.generate_signals(model, generation_cfg(model, cfg, ckpt, ev_spec),
                                   ev_spec.get("n_samples", 1000), ref.shape[-1], device, ref, seed)
        m = ev.quick_metrics(h_ref, ev.to_complex(fake), dt, seed=seed)
        row = {"trial": name, **{k: json.dumps(v) for k, v in params.items()},
               "train_min": round(train_min, 1), **{k: round(m[k], 5) for k in ev.SUMMARY_KEYS}}

        new_file = not os.path.exists(results_csv)
        with open(results_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new_file:
                w.writeheader()
            w.writerow(row)
        print(f"[{name}] score {m['selection_score']:.3f} | PDP {m['pdp_mae_db']:.2f} dB | "
              f"KS rms {m['rms_delay_ks']:.3f} | KS |h| {m['mag_ks']:.3f} | {train_min:.1f} min", flush=True)

    # Resumen ordenado
    if os.path.exists(results_csv):
        rows = sorted(csv.DictReader(open(results_csv)), key=lambda r: float(r["selection_score"]))
        cols = list(rows[0])
        with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
            f.write(f"Búsqueda {model.upper()} - referencia '{subset}' - ordenado por selection_score\n\n")
            f.write("".join(f"{c:>20}" for c in cols) + "\n")
            for r in rows:
                f.write("".join(f"{r[c]:>20}" for c in cols) + "\n")
        print(f"\nMejor prueba: {rows[0]['trial']} (score {rows[0]['selection_score']}). "
              f"Resumen en {out_dir}/summary.txt")


if __name__ == "__main__":
    main()
