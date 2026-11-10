"""
Normalización por tap para el entrenamiento de los modelos generativos.

El PDP de los canales loopback abarca más de 50 dB entre el primer tap y la cola. Con la
normalización global del dataset, los taps de la cola tienen amplitudes ~1e-3 y cualquier error
residual del modelo (p. ej. el ruido que queda tras el muestreo de difusión) se queda por encima
de ellos: aparece un "suelo de ruido" en el PDP generado.

TapScaler estandariza cada tap y cada componente (real/imag) por separado:
    z = (x - media[c, l]) / std[c, l] / escala
donde `escala` es el percentil 99.99 de |(x - media) / std| (~5 desviaciones típicas) y z se recorta
a [-1, 1] (compatible con la salida tanh de las GAN y con el clamp de los muestreadores de difusión).
Usar el máximo en vez del percentil desperdiciaría casi todo el rango: hay valores aislados a 25σ
(p. ej. la parte imaginaria del tap 0 de algún canal). Con el percentil solo se recorta ~0.01% de
los valores.
Así todos los taps tienen la misma importancia y el error del modelo se reparte en proporción
a la potencia de cada tap. `inverse` deshace la transformación (exacta salvo en los valores recortados).
"""
import os

import numpy as np
import torch

SCALER_FILE = "scaler.npz"


class TapScaler:
    def __init__(self, mean, std, scale):
        self.mean = np.asarray(mean, dtype=np.float32)  # (2, L)
        self.std = np.asarray(std, dtype=np.float32)    # (2, L)
        self.scale = float(scale)

    @classmethod
    def fit(cls, x, percentile=99.99, eps=1e-8):
        """x: (N, 2, L) dataset original."""
        x = np.asarray(x, dtype=np.float64)
        mean = x.mean(axis=0)
        std = x.std(axis=0) + eps
        scale = np.percentile(np.abs((x - mean) / std), percentile)
        return cls(mean, std, scale)

    @classmethod
    def identity(cls, L):
        return cls(np.zeros((2, L)), np.ones((2, L)), 1.0)

    def transform(self, x):
        return np.clip((x - self.mean) / self.std / self.scale, -1.0, 1.0).astype(np.float32)

    def inverse(self, z):
        """Acepta arrays de numpy o tensores (devuelve el mismo tipo)."""
        if torch.is_tensor(z):
            mean = torch.as_tensor(self.mean, device=z.device, dtype=z.dtype)
            std = torch.as_tensor(self.std, device=z.device, dtype=z.dtype)
            return z * self.scale * std + mean
        return (np.asarray(z) * self.scale * self.std + self.mean).astype(np.float32)

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        np.savez(os.path.join(directory, SCALER_FILE), mean=self.mean, std=self.std, scale=self.scale)

    @classmethod
    def load(cls, directory):
        """Carga el scaler guardado junto a un checkpoint; None si no existe (modelo sin normalizar)."""
        path = os.path.join(directory, SCALER_FILE)
        if not os.path.exists(path):
            return None
        d = np.load(path)
        return cls(d["mean"], d["std"], float(d["scale"]))


def prepare_training_data(config, save_dir):
    """
    Carga el dataset de la configuración y, si dataset.normalization == "per_tap", lo normaliza
    por tap y guarda el scaler en save_dir. Devuelve (datos_entrenamiento, datos_originales, scaler).
    """
    original = np.load(config["dataset"]["path"]).astype(np.float32)
    mode = config["dataset"].get("normalization", "none")
    if mode == "per_tap":
        scaler = TapScaler.fit(original)
        scaler.save(save_dir)
        data = scaler.transform(original)
        clipped = np.mean(np.abs(data) >= 1.0) * 100
        print(f"Normalización por tap (escala {scaler.scale:.2f}, {clipped:.4f}% de valores recortados); "
              f"scaler guardado en {save_dir}")
        return data, original, scaler
    if mode != "none":
        raise ValueError(f"normalization desconocida: {mode}")
    return original, original, TapScaler.identity(original.shape[-1])
