"""
Métrica de calidad para seleccionar el mejor checkpoint de las GAN y decidir el early stopping.

La pérdida del generador de una GAN no mide la calidad de los canales (depende de lo fuerte que
sea el discriminador en ese momento), así que el mejor modelo se elige comparando canales
generados con los reales:

    score = PDP_MAE_dB + 10 * KS_rms

  - PDP_MAE_dB: error medio en dB del perfil potencia-retardo, en los taps por encima de -60 dB.
  - KS_rms: estadístico de Kolmogorov-Smirnov (0..1) entre las distribuciones del RMS delay spread
    de cada canal. Detecta colapso de modos (canales generados todos parecidos), que el PDP medio
    no ve. El factor 10 hace que 0.1 de KS pese como 1 dB de error de PDP.

Menor es mejor.
"""
import numpy as np
import torch
from scipy.stats import ks_2samp


def _rms_delay_taps(h):
    p = np.abs(h) ** 2
    e = p.sum(axis=1) + 1e-30
    tau = np.arange(h.shape[1])
    mean = (p * tau).sum(axis=1) / e
    return np.sqrt((p * (tau[None] - mean[:, None]) ** 2).sum(axis=1) / e)


class ChannelQuality:
    def __init__(self, real, n_ref=5000, floor_db=-60.0, seed=0):
        """real: (N, 2, L) array o tensor con el dataset de entrenamiento."""
        real = real.cpu().numpy() if torch.is_tensor(real) else np.asarray(real)
        rng = np.random.default_rng(seed)
        real = real[rng.choice(len(real), min(n_ref, len(real)), replace=False)]
        h = real[:, 0].astype(np.float64) + 1j * real[:, 1]
        pdp = np.mean(np.abs(h) ** 2, axis=0)
        self.mask = pdp > pdp.max() * 10 ** (floor_db / 10)
        self.pdp_db = 10 * np.log10(pdp + 1e-30)
        self.rms = _rms_delay_taps(h)

    def __call__(self, fake):
        fake = fake.cpu().numpy() if torch.is_tensor(fake) else np.asarray(fake)
        h = fake[:, 0].astype(np.float64) + 1j * fake[:, 1]
        pdp_db = 10 * np.log10(np.mean(np.abs(h) ** 2, axis=0) + 1e-30)
        pdp_mae_db = float(np.mean(np.abs(pdp_db - self.pdp_db)[self.mask]))
        ks = float(ks_2samp(self.rms, _rms_delay_taps(h)).statistic)
        return {"score": pdp_mae_db + 10 * ks, "pdp_mae_db": pdp_mae_db, "rms_ks": ks}

    @torch.no_grad()
    def evaluate_generator(self, G, z_dim, device, n=2000, seed=1234, transform=None):
        """
        Genera n canales con G (en modo eval) y devuelve la métrica. Restaura el modo de G.
        transform: función que pasa la salida de G al espacio original (p. ej. TapScaler.inverse).
        """
        was_training = G.training
        G.eval()
        g = torch.Generator(device=device)
        g.manual_seed(seed)  # mismo ruido en todas las épocas: comparación justa
        z = torch.randn(n, z_dim, device=device, generator=g)
        x = G(z)
        out = self(transform(x) if transform is not None else x)
        G.train(was_training)
        return out
