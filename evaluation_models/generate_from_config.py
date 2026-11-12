"""Genera num_samples canales de un modelo con la configuración de evaluación indicada (uso: python evaluation_models/generate_from_config.py <config.json> <modelo>)."""
import sys, os, json, numpy as np
sys.path.insert(0, "evaluation_models"); sys.path.insert(0, ".")
import evaluation as ev
cfg_path, m = sys.argv[1], sys.argv[2]
cfg = json.load(open(cfg_path))
real = np.load(cfg["data"]["path"]).astype(np.float32)
mc = {**cfg["models"][m], "checkpoint": ev.repo_path(cfg["models"][m]["checkpoint"])}
seed = cfg["experiment"]["seed"]
ev.set_seed(seed)
data = ev.generate_signals(m, mc, cfg["evaluation"]["num_samples"], real.shape[-1], "cpu", real, seed,
                           batch_size=cfg["evaluation"]["batch_size"])
out = os.path.join(cfg["paths"]["save_dir"], "generated"); os.makedirs(out, exist_ok=True)
np.save(os.path.join(out, f"h_generated_{m}.npy"), data)
print("Generado", m, data.shape)
