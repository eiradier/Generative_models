"""
Cálculo de curvas de BER sobre canales en el dominio del tiempo (respuesta al impulso).

Modelo de enlace (OFDM con prefijo cíclico >= L, sin ISI):
    h[n] (L taps)  ->  H[k] = FFT_nfft(h)       (respuesta en frecuencia por subportadora)
    y[k] = H[k] x[k] + w[k],   w ~ CN(0, N0)
    x_hat[k] = y[k] / H[k]                       (ecualización ZF, canal perfectamente conocido)

Convenios:
- Símbolos con energía media Es = 1.
- Si normalize=True, cada canal se escala a energía unidad (sum |h|^2 = 1), de modo que
  E{|H[k]|^2} = 1 y la SNR es la SNR media recibida por subportadora: SNR = Es / N0.
  Así la BER solo depende de la forma del canal (selectividad en frecuencia) y no de su escala,
  que es lo que se quiere comparar entre canales reales y generados.
- Con el mismo `seed`, los bits y el ruido son idénticos para cualquier conjunto de canales
  del mismo tamaño, así que las diferencias de BER entre real y generado se deben solo al canal.
"""
import numpy as np
from scipy.special import erfc

BITS_PER_SYMBOL = {"QPSK": 2, "16QAM": 4}

# Gray 16QAM por eje: 00 -> -3, 01 -> -1, 11 -> +1, 10 -> +3
_PAM4_LEVELS = np.array([-3.0, -1.0, 3.0, 1.0])  # indexado por 2*b0 + b1


def modulate(bits, mod_type="QPSK"):
    """bits (n_sym * bps,) -> símbolos complejos con Es = 1 (mapeo Gray)."""
    if mod_type == "QPSK":
        b = bits.reshape(-1, 2)
        return ((1 - 2 * b[:, 0]) + 1j * (1 - 2 * b[:, 1])) / np.sqrt(2)
    if mod_type == "16QAM":
        b = bits.reshape(-1, 4)
        re = _PAM4_LEVELS[2 * b[:, 0] + b[:, 1]]
        im = _PAM4_LEVELS[2 * b[:, 2] + b[:, 3]]
        return (re + 1j * im) / np.sqrt(10)
    raise ValueError(f"Modulación no soportada: {mod_type}")


def demodulate(symbols, mod_type="QPSK"):
    """Decisión dura (Gray). Devuelve los bits en el mismo orden que `modulate`."""
    if mod_type == "QPSK":
        return np.stack([np.real(symbols) < 0, np.imag(symbols) < 0], axis=1).reshape(-1).astype(np.int8)
    if mod_type == "16QAM":
        s = symbols * np.sqrt(10)
        re, im = np.real(s), np.imag(s)
        return np.stack([re > 0, np.abs(re) < 2, im > 0, np.abs(im) < 2], axis=1).reshape(-1).astype(np.int8)
    raise ValueError(f"Modulación no soportada: {mod_type}")


def channel_frequency_response(h, nfft=None, normalize=True):
    """
    h: (N, L) complejo (respuesta al impulso, taps en el eje 1).
    Devuelve H: (N, nfft). Con normalize=True cada canal tiene energía unidad,
    por lo que mean_k |H[k]|^2 = 1 para cada canal.
    """
    h = np.asarray(h, dtype=np.complex128)
    if h.ndim != 2:
        raise ValueError("h debe tener forma (N, L)")
    N, L = h.shape
    nfft = L if nfft is None else int(nfft)
    if nfft < L:
        raise ValueError(f"nfft ({nfft}) debe ser >= número de taps ({L})")
    if normalize:
        h = h / (np.sqrt(np.sum(np.abs(h) ** 2, axis=1, keepdims=True)) + 1e-30)
    return np.fft.fft(h, n=nfft, axis=1)  # FFT a lo largo de los TAPS


def ber_curve(h, snr_db, mod_type="QPSK", nfft=None, normalize=True, n_ofdm=1, seed=0):
    """
    BER con ecualización ZF para una lista de SNR (dB).

    h        : (N, L) canales complejos en el tiempo.
    snr_db   : iterable de SNR medias por subportadora (Es/N0) en dB.
    nfft     : nº de subportadoras (>= L). Por defecto L.
    n_ofdm   : símbolos OFDM transmitidos por cada realización de canal.
    seed     : semilla de bits y ruido (la misma para real y generado).

    Devuelve un array con la BER para cada SNR.
    """
    H = channel_frequency_response(h, nfft=nfft, normalize=normalize)  # (N, K)
    N, K = H.shape
    bps = BITS_PER_SYMBOL[mod_type]

    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=n_ofdm * N * K * bps, dtype=np.int8)
    x = modulate(bits, mod_type).reshape(n_ofdm, N, K)
    w_unit = (rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape)) / np.sqrt(2)

    Hx = H[None] * x
    H_safe = np.where(np.abs(H) > 1e-15, H, 1e-15)[None]

    bers = []
    for snr in np.atleast_1d(snr_db):
        n0 = 10 ** (-snr / 10)  # Es = 1
        y = Hx + np.sqrt(n0) * w_unit
        x_hat = y / H_safe
        rx_bits = demodulate(x_hat.reshape(-1), mod_type)
        bers.append(np.mean(rx_bits != bits))
    return np.array(bers)


def compute_ber(h, snr_db=10.0, mod_type="QPSK", **kwargs):
    """BER para una única SNR (compatibilidad con la versión anterior)."""
    return float(ber_curve(h, [snr_db], mod_type=mod_type, **kwargs)[0])


# Curvas teóricas (Gray, SNR = Es/N0 en dB)

def ber_theory_awgn(snr_db, mod_type="QPSK"):
    g = 10 ** (np.asarray(snr_db, dtype=float) / 10)
    if mod_type == "QPSK":
        return 0.5 * erfc(np.sqrt(g / 2))
    if mod_type == "16QAM":  # aproximación de vecinos más cercanos
        return 3 / 8 * erfc(np.sqrt(g / 10))
    raise ValueError(mod_type)


def ber_theory_rayleigh(snr_db, mod_type="QPSK"):
    """Rayleigh plano por subportadora con ZF (E|H|^2 = 1)."""
    g = 10 ** (np.asarray(snr_db, dtype=float) / 10)
    if mod_type == "QPSK":
        a = g / 2
        return 0.5 * (1 - np.sqrt(a / (1 + a)))
    if mod_type == "16QAM":
        a = g / 10
        return 3 / 8 * (1 - np.sqrt(a / (1 + a)))
    raise ValueError(mod_type)
