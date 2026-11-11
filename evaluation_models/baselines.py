"""
Modelos estadísticos clásicos de referencia para comparar con los modelos generativos.
Se ajustan con el conjunto de entrenamiento (los mismos canales con los que se entrenan las redes).

- TDLRice ("tdl"): modelo de línea de retardos (TDL) con dispersión incorrelada (WSSUS). Cada tap es
  independiente y sigue una distribución Rice: h_l = mu_l + CN(0, sigma_l^2), donde mu_l es la componente
  fija (media compleja del tap) y sigma_l^2 la potencia de la parte difusa. Reproduce exactamente el PDP
  y el factor K de cada tap, pero no la correlación entre taps.
- GaussianFull ("gauss"): vector gaussiano de dimensión 2L con la media y la matriz de covarianza completa
  de [Re(h), Im(h)] (incluye la correlación entre taps y la impropiedad). Es el mejor modelo posible con
  estadística de segundo orden: lo que los modelos generativos mejoren sobre él se debe a estadística
  de orden superior (no gaussiana).

En el preprocesado todos los canales se normalizan a la misma energía, así que las muestras de ambos
modelos se reescalan a esa energía (la de los datos de entrenamiento), igual que los canales reales.
"""
import numpy as np

BASELINES = ("tdl", "gauss")


def _to_complex(x):
    return x[:, 0, :].astype(np.float64) + 1j * x[:, 1, :].astype(np.float64)


def _to_real(h):
    return np.stack([h.real, h.imag], axis=1).astype(np.float32)


def _normalize_energy(h, energy):
    e = np.sum(np.abs(h) ** 2, axis=1, keepdims=True)
    return h * np.sqrt(energy / np.maximum(e, 1e-30))


class TDLRice:
    def fit(self, x):
        h = _to_complex(x)
        self.mu = h.mean(axis=0)
        self.sigma2 = np.mean(np.abs(h - self.mu) ** 2, axis=0)
        self.energy = float(np.mean(np.sum(np.abs(h) ** 2, axis=1)))
        return self

    def k_factor_db(self):
        """Factor K de Rice de cada tap: |mu|^2 / sigma^2 (dB)."""
        return 10 * np.log10(np.abs(self.mu) ** 2 / np.maximum(self.sigma2, 1e-30) + 1e-30)

    def sample(self, n, rng):
        L = len(self.mu)
        w = (rng.standard_normal((n, L)) + 1j * rng.standard_normal((n, L))) * np.sqrt(self.sigma2 / 2)
        return _to_real(_normalize_energy(self.mu + w, self.energy))


class GaussianFull:
    def fit(self, x):
        h = _to_complex(x)
        v = np.concatenate([h.real, h.imag], axis=1)  # (N, 2L)
        self.L = h.shape[1]
        self.mean = v.mean(axis=0)
        cov = np.cov(v, rowvar=False)
        w, U = np.linalg.eigh(cov)
        self.A = U * np.sqrt(np.clip(w, 0, None))  # cov = A A^T
        self.energy = float(np.mean(np.sum(np.abs(h) ** 2, axis=1)))
        return self

    def sample(self, n, rng):
        v = self.mean + rng.standard_normal((n, 2 * self.L)) @ self.A.T
        h = v[:, :self.L] + 1j * v[:, self.L:]
        return _to_real(_normalize_energy(h, self.energy))


def fit_baseline(name, train_data):
    return {"tdl": TDLRice, "gauss": GaussianFull}[name]().fit(train_data)


def generate_baseline(name, train_data, num_samples, seed):
    """Ajusta el modelo `name` con train_data (N, 2, L) y genera num_samples canales (num_samples, 2, L)."""
    return fit_baseline(name, train_data).sample(num_samples, np.random.default_rng(seed))
