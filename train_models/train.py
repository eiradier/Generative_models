"""
Lanza el entrenamiento de uno o varios modelos.

Uso (desde la raíz del repositorio):
    python train_models/train.py --model gan
    python train_models/train.py --model gan dcgan ddpm ddim
    python train_models/train.py --model all

Cada modelo usa su archivo de configuración (models_configs/gans_config.json o dm_config.json)
y guarda su mejor checkpoint en <save_dir>/<MODELO>/ (p. ej. checkpoints/sodupe_180/DDIM/).
Si hay GPU (CUDA) se usa automáticamente.
"""
import argparse
import importlib
import os
import sys
import time

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, REPO_ROOT)

MODELS = {
    "gan": ("train_models.train_GAN_Conv1D", "gans_config.json"),
    "dcgan": ("train_models.train_DCGAN_Conv1D", "gans_config.json"),
    "wgan": ("train_models.train_WGAN", "gans_config.json"),
    "ddpm": ("train_models.train_DDPM", "dm_config.json"),
    "ddim": ("train_models.train_DDIM", "dm_config.json"),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento de modelos generativos de canal")
    parser.add_argument("--model", nargs="+", required=True, choices=list(MODELS) + ["all"])
    parser.add_argument("--config", default=None,
                        help="Configuración alternativa (por defecto la de models_configs/)")
    args = parser.parse_args()

    os.chdir(REPO_ROOT)  # las rutas de las configuraciones son relativas a la raíz
    models = list(MODELS) if "all" in args.model else args.model

    for name in models:
        module_name, cfg = MODELS[name]
        cfg_path = args.config or os.path.join(THIS_DIR, "models_configs", cfg)
        print(f"\n===== Entrenando {name.upper()} con {cfg_path} =====")
        t0 = time.time()
        importlib.import_module(module_name).train(cfg_path)
        print(f"===== {name.upper()} terminado en {(time.time() - t0) / 60:.1f} min =====")
