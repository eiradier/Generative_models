"""
Evaluación de los modelos generativos de canal (GAN, DCGAN, WGAN, DDPM, DDIM) frente a
las medidas reales de loopback.

Para cada modelo:
  1. Genera `num_samples` canales con el checkpoint entrenado (o reutiliza los ya generados).
  2. Calcula métricas frente al dataset real: PDP, retardo medio y RMS delay spread (sin umbral y
     con umbrales respecto al pico de cada canal, evaluation.delay_thresholds_db), distribución de |h|,
     potencia media de la respuesta en frecuencia, función de correlación en frecuencia, ancho de
     banda de coherencia y curvas de BER (QPSK/16QAM con ZF).
Después genera las figuras comparativas (todos los modelos juntos, estilo artículo) y las tablas.

Uso (desde la raíz del repositorio):
    python evaluation_models/evaluation.py
    python evaluation_models/evaluation.py --models ddim wgan --device cuda
    python evaluation_models/evaluation.py --reuse-generated      # no vuelve a generar

Resultados en paths.save_dir (evaluation_config.json):
    generated/h_generated_<modelo>.npy   canales generados (N, 2, L)
    metrics.pkl                          todas las métricas (arrays incluidos)
    metrics_summary.txt                  tabla de métricas escalares
    ber_table.txt                        BER por modulación y SNR
    fig_*.pdf / fig_*.png                figuras
"""
import argparse
import importlib
import json
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import gaussian_kde, kurtosis, ks_2samp, ttest_ind

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, REPO_ROOT)

from BER import ber_curve, ber_theory_awgn, ber_theory_rayleigh  # noqa: E402
from baselines import BASELINES, generate_baseline  # noqa: E402
from train_models.normalization import TapScaler  # noqa: E402

MODEL_LABELS = {"gan": "GAN", "dcgan": "DCGAN", "wgan": "WGAN", "ddpm": "DDPM", "ddim": "DDIM",
                "tdl": "TDL-Rice", "gauss": "Gaussian"}
COLORS = {"gan": "red", "dcgan": "blue", "wgan": "purple", "ddpm": "green", "ddim": "orange",
          "tdl": "teal", "gauss": "saddlebrown"}
# Modelos clásicos de referencia (baselines.py) en línea de puntos y rayas
LINESTYLES = {"tdl": "-.", "gauss": "-."}

MODULES = {
    "gan": "train_models.train_GAN_Conv1D",
    "dcgan": "train_models.train_DCGAN_Conv1D",
    "wgan": "train_models.train_WGAN",
    "ddpm": "train_models.train_DDPM",
    "ddim": "train_models.train_DDIM",
}


# Utilidades

def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_db(x, eps=1e-12):
    return 10 * np.log10(np.maximum(x, eps))


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def mse(a, b):
    return float(np.mean(np.abs(a - b) ** 2))


def to_complex(data):
    """(N, 2, L) [real, imag] -> (N, L) complejo."""
    return data[:, 0, :].astype(np.float64) + 1j * data[:, 1, :].astype(np.float64)


def repo_path(p):
    """Rutas relativas de la configuración -> relativas a la raíz del repositorio."""
    return p if os.path.isabs(p) else os.path.join(REPO_ROOT, p)


def sampling_period(config):
    """Periodo de muestreo de los taps. DVB-T2: fs = 8/7 * BW (8 MHz -> 109.375 ns)."""
    s = config.get("sampling", {})
    fs = s.get("fs_hz") or 8.0 / 7.0 * s.get("bandwidth_hz", 8e6)
    return 1.0 / fs


# Generación de canales

def _load_checkpoint(path, device):
    return torch.load(path, map_location=device, weights_only=True)


@torch.no_grad()
def generate_signals(model_type, model_cfg, num_samples, L, device, real_data, seed, batch_size=500):
    """
    Genera `num_samples` canales con el modelo entrenado. Devuelve (num_samples, 2, L) float32.
    El z_dim de las GAN se lee del propio checkpoint, para que no pueda quedar desalineado
    con la configuración usada al entrenar.
    """
    m = importlib.import_module(MODULES[model_type])
    ckpt = _load_checkpoint(model_cfg["checkpoint"], device)
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    out = []

    if model_type in ("gan", "dcgan", "wgan"):
        state = ckpt["G"]
        z_dim = state["fc.weight"].shape[1]
        G = m.Generator(z_dim=z_dim, L=L).to(device)
        G.load_state_dict(state)
        G.eval()
        for i in range(0, num_samples, batch_size):
            b = min(batch_size, num_samples - i)
            z = torch.randn(b, z_dim, device=device, generator=g)
            out.append(G(z).cpu().numpy())

    elif model_type == "ddpm":
        unet = m.UNet1D(time_emb_dim=model_cfg.get("time_emb_dim", 64)).to(device)
        unet.load_state_dict(ckpt)
        unet.eval()
        ddpm = m.DDPM(unet, T=model_cfg["T"]).to(device)
        clip_x0 = model_cfg.get("clip_x0", True)
        betas, abar = ddpm.betas, ddpm.alphas_cumprod
        abar_prev = torch.cat([torch.ones(1, device=abar.device), abar[:-1]])
        for i in range(0, num_samples, batch_size):
            b = min(batch_size, num_samples - i)
            x = torch.randn(b, 2, L, device=device, generator=g)
            for t in reversed(range(ddpm.T)):
                t_batch = torch.full((b,), t, device=device, dtype=torch.long)
                eps = ddpm.model(x, t_batch)
                if clip_x0:
                    # Muestreo DDPM estándar con x0 recortado ("clip_denoised"): se estima x0, se
                    # recorta al rango de los datos y se usa la media del posterior q(x_{t-1}|x_t,x0).
                    # Sin el recorte, con el schedule coseno (beta_T ~ 0.999) el primer paso divide
                    # por sqrt(1 - beta) ~ 0.03 y amplifica el error del modelo.
                    x0 = (x - torch.sqrt(1 - abar[t]) * eps) / torch.sqrt(abar[t])
                    x0 = torch.clamp(x0, -1, 1)
                    mean = (betas[t] * torch.sqrt(abar_prev[t]) / (1 - abar[t])) * x0 \
                        + ((1 - abar_prev[t]) * torch.sqrt(1 - betas[t]) / (1 - abar[t])) * x
                    var = betas[t] * (1 - abar_prev[t]) / (1 - abar[t])
                else:
                    mean = (x - betas[t] / torch.sqrt(1 - abar[t]) * eps) / torch.sqrt(1 - betas[t])
                    var = betas[t]
                x = mean
                if t > 0:
                    x = x + torch.sqrt(var) * torch.randn(x.shape, device=device, generator=g)
            out.append(torch.clamp(x, -1, 1).cpu().numpy())

    elif model_type == "ddim":
        unet = m.UNet1D(time_emb_dim=model_cfg.get("time_emb_dim", 64)).to(device)
        unet.load_state_dict(ckpt)
        unet.eval()
        diffusion = m.Diffusion(unet, T=model_cfg["T"], data=real_data).to(device)  # data solo afecta a la loss
        sampler = m.DDIMSampler(diffusion, eta=model_cfg.get("eta", 0.0))
        torch.manual_seed(seed)  # el sampler usa el generador global
        for i in range(0, num_samples, batch_size):
            b = min(batch_size, num_samples - i)
            out.append(sampler.sample(shape=(b, 2, L), device=device,
                                      steps=model_cfg.get("sampling_steps", 100)).cpu().numpy())
    else:
        raise ValueError(model_type)

    fake = np.concatenate(out, axis=0).astype(np.float32)
    # Si el modelo se entrenó con normalización por tap, deshacerla (scaler.npz junto al checkpoint)
    scaler = TapScaler.load(os.path.dirname(model_cfg["checkpoint"]))
    if scaler is not None:
        fake = scaler.inverse(fake)
    return fake


# Métricas

def delay_moments(p, tau, threshold_db=None):
    """
    Retardo medio y RMS delay spread de cada canal (p: |h|^2, (N, L)).
    threshold_db: si se indica, solo cuentan los taps con potencia >= pico del canal - threshold_db
    (el resto se pone a 0). None = todos los taps (sin umbral).
    """
    if threshold_db is not None:
        p = np.where(p >= p.max(axis=1, keepdims=True) * 10 ** (-threshold_db / 10), p, 0.0)
    e = p.sum(axis=1) + 1e-30
    mean_delay = (p * tau).sum(axis=1) / e
    rms_delay = np.sqrt((p * (tau[None] - mean_delay[:, None]) ** 2).sum(axis=1) / e)
    return mean_delay, rms_delay


def threshold_tag(thr):
    """Sufijo de las claves/figuras para cada umbral del delay spread ('' = sin umbral)."""
    return "" if thr is None else f"_th{thr:g}dB"


def threshold_label(thr):
    return "sin umbral" if thr is None else f"umbral {thr:g} dB"


# Parámetros por canal (clásicos, no de segundo orden). Configurables en evaluation.channel_params.
CHANNEL_PARAM_DEFAULTS = {"fade_threshold_db": 10.0, "se_snr_db": 20.0, "delay_window_pct": 90.0,
                          "bc_threshold": 0.9}


def per_channel_params(h, p, Hn, dt, delay_thresholds, cfg):
    """
    Parámetros clásicos calculados canal a canal (cada uno es un array de N valores):
      ripple_db          rizado de la respuesta en frecuencia, max_f |H|^2 / min_f |H|^2 (dB)
      fade_frac          fracción de la banda con |H(f)|^2 < media - fade_threshold_db
      se                 eficiencia espectral media en la banda a se_snr_db, E_f{log2(1 + SNR |H(f)|^2)}
                         (bit/s/Hz; su percentil 10 es la eficiencia de outage al 10%)
      bc_inst_mhz        ancho de banda de coherencia de cada canal: primer Δf con |R_H(Δf)| < bc_threshold,
                         con R_H la DFT del perfil |h|^2 del propio canal (MHz)
      delay_window_ns    ventana de retardo (ITU-R P.1407): anchura del intervalo central que contiene el
                         delay_window_pct % de la energía (ns)
      n_taps_energy      número mínimo de taps que reúnen el delay_window_pct % de la energía
      delay_interval_th<X>dB  intervalo de retardo (ITU-R P.1407): del primer al último tap por encima de
                         pico - X dB, para cada umbral de delay_thresholds (ns)
    Hn: |H(f)|^2 normalizado a su media en la banda (N, L).
    """
    N, L = h.shape
    q = cfg["delay_window_pct"] / 100.0
    out = {}
    out["ripple_db"] = 10 * np.log10(Hn.max(axis=1) / np.maximum(Hn.min(axis=1), 1e-30))
    out["fade_frac"] = np.mean(Hn < 10 ** (-cfg["fade_threshold_db"] / 10), axis=1)
    out["se"] = np.mean(np.log2(1 + 10 ** (cfg["se_snr_db"] / 10) * Hn), axis=1)

    R = np.abs(np.fft.fft(p, axis=1))[:, : L // 2 + 1]
    R = R / np.maximum(R[:, :1], 1e-30)
    below = R < cfg["bc_threshold"]
    i = np.where(below.any(axis=1), below.argmax(axis=1), L // 2)
    i0 = np.maximum(i - 1, 0)
    r1, r2 = R[np.arange(N), i0], R[np.arange(N), i]
    frac = np.where(i > 0, (r1 - cfg["bc_threshold"]) / np.maximum(r1 - r2, 1e-30), 0.0)
    out["bc_inst_mhz"] = (i0 + frac) / (dt * L) / 1e6

    ce = np.cumsum(p, axis=1) / np.maximum(p.sum(axis=1, keepdims=True), 1e-30)
    lo, hi = (1 - q) / 2, 1 - (1 - q) / 2
    out["delay_window_ns"] = ((ce >= hi).argmax(axis=1) - (ce >= lo).argmax(axis=1)) * dt * 1e9
    cs = np.cumsum(np.sort(p, axis=1)[:, ::-1], axis=1) / np.maximum(p.sum(axis=1, keepdims=True), 1e-30)
    out["n_taps_energy"] = (cs < q).sum(axis=1) + 1

    for thr in delay_thresholds:
        if thr is None:
            continue
        above = p >= p.max(axis=1, keepdims=True) * 10 ** (-thr / 10)
        first = above.argmax(axis=1)
        last = L - 1 - above[:, ::-1].argmax(axis=1)
        out[f"delay_interval{threshold_tag(thr)}"] = (last - first) * dt * 1e9
    return out


def channel_param_keys(delay_thresholds):
    keys = ["ripple_db", "fade_frac", "se", "bc_inst_mhz", "delay_window_ns", "n_taps_energy"]
    return keys + [f"delay_interval{threshold_tag(t)}" for t in delay_thresholds if t is not None]


def channel_stats(h, dt, mag_grid, mag_db_grid, kde_points=50000, seed=0, delay_thresholds=(None,),
                  channel_params=None):
    """
    Estadísticas de un conjunto de canales complejos h (N, L).
    delay_thresholds: umbrales (dB bajo el pico de cada canal) para el retardo medio y el RMS delay
    spread; None = sin umbral. Las claves sin sufijo (mean_delay, rms_delay, ...) son siempre las de
    sin umbral; las demás llevan el sufijo threshold_tag(thr).
    """
    N, L = h.shape
    p = np.abs(h) ** 2
    energy = p.sum(axis=1)
    tau = np.arange(L) * dt

    mean_delay, rms_delay = delay_moments(p, tau)
    delays = {}
    for thr in delay_thresholds:
        if thr is None:
            continue
        md, rd = delay_moments(p, tau, thr)
        tag = threshold_tag(thr)
        delays.update({f"mean_delay{tag}_all": md, f"rms_delay{tag}_all": rd,
                       f"mean_delay{tag}": float(md.mean()), f"rms_delay{tag}": float(rd.mean())})

    pdp = p.mean(axis=0)

    # Respuesta en frecuencia: potencia media por subportadora
    H = np.fft.fft(h, axis=1)
    freq_power = np.fft.fftshift((np.abs(H) ** 2).mean(axis=0))

    # Nulos en frecuencia de cada canal (|H(f)|^2 normalizado a su media en la banda):
    # profundidad del nulo más profundo y ganancia de ruido del ecualizador ZF, E_f{1/|H(f)|^2}, que es lo
    # que determina la BER con ZF. Son funciones no lineales de h: un modelo que solo reproduzca la
    # estadística de segundo orden (p. ej. el gaussiano) no tiene por qué acertarlas.
    Hn = np.abs(H) ** 2
    Hn = Hn / (Hn.mean(axis=1, keepdims=True) + 1e-30)
    null_depth_db = 10 * np.log10(Hn.min(axis=1) + 1e-30)
    zf_gain_db = 10 * np.log10(np.mean(1.0 / np.maximum(Hn, 1e-12), axis=1))

    # Curtosis en exceso de la parte difusa de cada tap (h - media del tap), Re e Im por separado
    # (0 = gaussiana; < 0: distribución más "plana", p. ej. fase que rota; > 0: colas pesadas).
    dif = h - h.mean(axis=0)
    kurt_tap = 0.5 * (kurtosis(dif.real, axis=0) + kurtosis(dif.imag, axis=0))

    cp_cfg = {**CHANNEL_PARAM_DEFAULTS, **(channel_params or {})}
    params = per_channel_params(h, p, Hn, dt, delay_thresholds, cp_cfg)

    # Función de correlación en frecuencia R(Δk) = E{H[k+Δk] H*[k]} (circular, promediada en k
    # y en realizaciones). Por la relación de Wiener-Khinchin es la DFT del PDP medio.
    R = np.fft.fft(pdp)
    freq_corr = np.abs(R[: L // 2 + 1]) / (np.abs(R[0]) + 1e-30)

    mag = np.abs(h).ravel()
    mag_db = 20 * np.log10(mag + 1e-12)
    mag_sorted = np.sort(mag)
    mag_db_sorted = np.sort(mag_db)
    cdf = np.searchsorted(mag_sorted, mag_grid, side="right") / mag.size
    cdf_db = np.searchsorted(mag_db_sorted, mag_db_grid, side="right") / mag.size

    rng = np.random.default_rng(seed)
    sub = rng.choice(mag.size, min(kde_points, mag.size), replace=False)
    kde = gaussian_kde(mag[sub])(mag_grid)
    kde_db = gaussian_kde(mag_db[sub])(mag_db_grid)

    return {
        "pdp": pdp, "pdp_db": to_db(pdp),
        "energy": energy,
        "mean_delay_all": mean_delay, "rms_delay_all": rms_delay,
        "mean_delay": float(mean_delay.mean()), "rms_delay": float(rms_delay.mean()),
        "std_tap": np.abs(h).std(axis=0),
        "freq_power": freq_power,
        "null_depth_db_all": null_depth_db, "zf_gain_db_all": zf_gain_db,
        "kurt_tap": kurt_tap,
        "channel_params": cp_cfg,
        **{f"{k}_all": v for k, v in params.items()},
        "freq_corr": freq_corr,
        "mag_cdf": cdf, "mag_cdf_db": cdf_db,
        "mag_kde": kde, "mag_kde_db": kde_db,
        "_mag_sub": mag[sub],
        "delay_thresholds": list(delay_thresholds),
        **delays,
    }


def coherence_bandwidth(freq_corr, df, threshold):
    """Primer desplazamiento en frecuencia donde |R| cae por debajo del umbral (interpolado). Hz."""
    below = np.where(freq_corr < threshold)[0]
    if len(below) == 0:
        return np.nan  # no cae por debajo del umbral en el rango disponible
    i = below[0]
    if i == 0:
        return 0.0
    y1, y2 = freq_corr[i - 1], freq_corr[i]
    return float((i - 1 + (y1 - threshold) / (y1 - y2 + 1e-30)) * df)


def compare(real, fake, df, thresholds):
    """Métricas escalares de un modelo frente al real."""
    m = {}
    mask = real["pdp"] > real["pdp"].max() * 1e-6  # taps por encima de -60 dB
    m["pdp_mae"] = mae(real["pdp"], fake["pdp"])
    m["pdp_mse"] = mse(real["pdp"], fake["pdp"])
    m["pdp_mae_db"] = mae(real["pdp_db"][mask], fake["pdp_db"][mask])

    m["energy_ratio_db"] = float(10 * np.log10(fake["energy"].mean() / real["energy"].mean()))

    tags = [threshold_tag(t) for t in real.get("delay_thresholds", [None]) if t is not None]
    for k in [f"{base}{tag}" for tag in [""] + tags for base in ("mean_delay", "rms_delay")]:
        m[f"{k}_real_ns"] = real[k] * 1e9
        m[f"{k}_fake_ns"] = fake[k] * 1e9
        m[f"{k}_mae_ns"] = abs(real[k] - fake[k]) * 1e9
        m[f"{k}_ttest_p"] = float(ttest_ind(real[f"{k}_all"], fake[f"{k}_all"], equal_var=False).pvalue)
        ks = ks_2samp(real[f"{k}_all"], fake[f"{k}_all"])
        m[f"{k}_ks"], m[f"{k}_ks_p"] = float(ks.statistic), float(ks.pvalue)

    m["std_tap_mae"] = mae(real["std_tap"], fake["std_tap"])
    m["mag_ks"] = float(ks_2samp(real["_mag_sub"], fake["_mag_sub"]).statistic)

    ref = real["freq_power"].max()
    m["freq_power_mae_db"] = mae(to_db(real["freq_power"] / ref), to_db(fake["freq_power"] / ref))
    m["freq_corr_mae"] = mae(real["freq_corr"], fake["freq_corr"])

    for k in ("null_depth_db", "zf_gain_db"):
        m[f"{k}_real"] = float(np.median(real[f"{k}_all"]))
        m[f"{k}_fake"] = float(np.median(fake[f"{k}_all"]))
        m[f"{k}_ks"] = float(ks_2samp(real[f"{k}_all"], fake[f"{k}_all"]).statistic)
    # Curtosis: el tap del pico (acoplamiento directo, casi constante) va aparte porque tiene valores
    # atípicos esporádicos que la disparan (decenas). El resto: taps a menos de 30 dB del pico del PDP
    # real (en la cola la estimación es ruidosa); mediana y MAE.
    peak = int(np.argmax(real["pdp"]))
    strong = real["pdp"] > real["pdp"].max() * 1e-3
    strong[peak] = False
    m["kurt_peak_real"] = float(real["kurt_tap"][peak])
    m["kurt_peak_fake"] = float(fake["kurt_tap"][peak])
    m["kurt_real"] = float(np.median(real["kurt_tap"][strong]))
    m["kurt_fake"] = float(np.median(fake["kurt_tap"][strong]))
    m["kurt_mae"] = mae(real["kurt_tap"][strong], fake["kurt_tap"][strong])

    # Parámetros por canal: mediana real/generada y KS
    for k in channel_param_keys(real.get("delay_thresholds", [None])):
        m[f"{k}_real"] = float(np.median(real[f"{k}_all"]))
        m[f"{k}_fake"] = float(np.median(fake[f"{k}_all"]))
        m[f"{k}_ks"] = float(ks_2samp(real[f"{k}_all"], fake[f"{k}_all"]).statistic)
    m["se_out10_real"] = float(np.percentile(real["se_all"], 10))
    m["se_out10_fake"] = float(np.percentile(fake["se_all"], 10))
    for thr in thresholds:
        bc_r = coherence_bandwidth(real["freq_corr"], df, thr)
        bc_f = coherence_bandwidth(fake["freq_corr"], df, thr)
        m[f"bc{int(thr * 100)}_real_mhz"] = bc_r / 1e6
        m[f"bc{int(thr * 100)}_fake_mhz"] = bc_f / 1e6
        m[f"bc{int(thr * 100)}_err_mhz"] = abs(bc_r - bc_f) / 1e6
    return m


def quick_metrics(h_ref, h_fake, dt, thresholds=(0.9, 0.5), ber_snr=(5, 15, 25), seed=0):
    """
    Resumen de la calidad de unos canales generados frente a una referencia (p. ej. validación),
    para comparar configuraciones. Devuelve un dict con métricas escalares y `selection_score`:

        selection_score = PDP_MAE_dB + 10 * (KS_rms_delay + KS_|h|)

    (menor es mejor). Combina el ajuste del PDP medio con el de dos distribuciones por canal
    (RMS delay spread y magnitud), que detectan colapso de modos y suelos de ruido.
    """
    L = h_ref.shape[1]
    allmag = np.concatenate([np.abs(h_ref).ravel(), np.abs(h_fake).ravel()])
    mag_grid = np.linspace(0, allmag.max(), 300)
    mag_db_grid = np.linspace(-80, 20 * np.log10(allmag.max() + 1e-12), 300)
    r = channel_stats(h_ref, dt, mag_grid, mag_db_grid, kde_points=20000, seed=seed)
    f = channel_stats(h_fake, dt, mag_grid, mag_db_grid, kde_points=20000, seed=seed)
    m = compare(r, f, 1 / dt / L, thresholds)
    n = min(len(h_ref), len(h_fake))
    b_ref = ber_curve(h_ref[:n], list(ber_snr), "QPSK", seed=seed)
    b_fake = ber_curve(h_fake[:n], list(ber_snr), "QPSK", seed=seed)
    m["ber_qpsk_dlog10"] = float(np.mean(np.abs(np.log10(np.maximum(b_fake, 1e-12))
                                                - np.log10(np.maximum(b_ref, 1e-12)))))
    m["selection_score"] = m["pdp_mae_db"] + 10 * (m["rms_delay_ks"] + m["mag_ks"])
    return m


SUMMARY_KEYS = ["selection_score", "pdp_mae_db", "rms_delay_ks", "mag_ks", "mean_delay_ks",
                "rms_delay_fake_ns", "freq_corr_mae", "bc50_err_mhz", "ber_qpsk_dlog10"]


# Figuras

def _style(ax, xlabel, ylabel, legend=True, loc="best"):
    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=10)
    if legend:
        ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=10, loc=loc)


def _save(fig, save_dir, name):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(save_dir, f"{name}.{ext}"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def _plot_curves(ax, x, real_y, fakes, key, transform=lambda v: v):
    for name, st in fakes.items():
        ax.plot(x, transform(st[key]), label=MODEL_LABELS[name], color=COLORS[name], lw=1.3,
                ls=LINESTYLES.get(name, "--"))
    ax.plot(x, transform(real_y), label="Real", color="black", lw=2.0)


def plot_all(save_dir, real, fakes, grids, dt, thresholds, ber_results, ber_cfg, examples):
    L = len(real["pdp"])
    delay_us = np.arange(L) * dt * 1e6
    fs = 1 / dt
    df = fs / L

    # PDP lineal y en dB
    fig, ax = plt.subplots(figsize=(9, 5))
    _plot_curves(ax, delay_us, real["pdp"], fakes, "pdp")
    _style(ax, r"Delay $\tau$ ($\mu$s)", "Power Delay Profile (linear)")
    _save(fig, save_dir, "fig_pdp_linear")

    fig, ax = plt.subplots(figsize=(9, 5))
    _plot_curves(ax, delay_us, real["pdp_db"], fakes, "pdp_db")
    _style(ax, r"Delay $\tau$ ($\mu$s)", "Power Delay Profile (dB)")
    _save(fig, save_dir, "fig_pdp_db")

    # Distribución de |h|: KDE y CDF (lineal, con zoom) y CDF en dB
    x, x_db = grids
    fig, ax = plt.subplots(figsize=(9, 5))
    _plot_curves(ax, x, real["mag_kde"], fakes, "mag_kde")
    _style(ax, r"Signal Magnitude $|h[n]|$", "Probability Density")
    _save(fig, save_dir, "fig_kde_comparison")

    fig, ax = plt.subplots(figsize=(9, 6))
    _plot_curves(ax, x, real["mag_cdf"], fakes, "mag_cdf")
    _style(ax, r"Signal Magnitude $|h[n]|$", r"$P(|h[n]| \leq x)$", loc="lower right")
    axins = ax.inset_axes([0.3, 0.3, 0.40, 0.35])
    for name, st in fakes.items():
        axins.plot(x, st["mag_cdf"], color=COLORS[name], lw=1.1, ls=LINESTYLES.get(name, "--"))
    axins.plot(x, real["mag_cdf"], color="black", lw=1.5)
    axins.set_xlim(x[int(len(x) * 0.04)], x[int(len(x) * 0.12)])
    axins.set_ylim(0.90, 1.02)
    axins.grid(True, linestyle=":", alpha=0.6)
    axins.tick_params(labelsize=9)
    ax.indicate_inset_zoom(axins, edgecolor="gray", alpha=0.8)
    _save(fig, save_dir, "fig_cdf_comparison")

    fig, ax = plt.subplots(figsize=(9, 5))
    _plot_curves(ax, x_db, real["mag_cdf_db"], fakes, "mag_cdf_db")
    _style(ax, r"Signal Magnitude $20\log_{10}|h[n]|$ (dB)", r"$P(|h[n]| \leq x)$", loc="upper left")
    _save(fig, save_dir, "fig_cdf_comparison_db")

    # Potencia media de la respuesta en frecuencia (referida al máximo del real)
    f_mhz = np.fft.fftshift(np.fft.fftfreq(L, d=dt)) / 1e6
    ref = real["freq_power"].max()
    fig, ax = plt.subplots(figsize=(9, 5))
    _plot_curves(ax, f_mhz, real["freq_power"], fakes, "freq_power", transform=lambda v: to_db(v / ref))
    _style(ax, "Frequency (MHz)", r"$E\{|H(f)|^2\}$ (dB)")
    _save(fig, save_dir, "fig_freq_response_db")

    # Correlación en frecuencia y ancho de banda de coherencia
    lag_mhz = np.arange(L // 2 + 1) * df / 1e6
    fig, ax = plt.subplots(figsize=(9, 5))
    _plot_curves(ax, lag_mhz, real["freq_corr"], fakes, "freq_corr")
    for thr in thresholds:
        ax.axhline(thr, color="gray", lw=0.8, ls=":")
    _style(ax, r"Frequency separation $\Delta f$ (MHz)", r"$|R_H(\Delta f)|$ (normalized)")
    _save(fig, save_dir, "fig_frequency_correlation")

    # Nulos en frecuencia: CDF de la profundidad del nulo y de la ganancia de ruido ZF
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, key, xl in ((axes[0], "null_depth_db_all", r"Deepest null $\min_f |H(f)|^2$ (dB)"),
                        (axes[1], "zf_gain_db_all", r"ZF noise gain $E_f\{1/|H(f)|^2\}$ (dB)")):
        for n, st in fakes.items():
            v = np.sort(st[key])
            ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=COLORS[n], ls=LINESTYLES.get(n, "--"), lw=1.3,
                    label=MODEL_LABELS[n])
        v = np.sort(real[key])
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), color="black", lw=2, label="Real")
        _style(ax, xl, "CDF", legend=ax is axes[0], loc="lower right" if ax is axes[1] else "upper left")
    _save(fig, save_dir, "fig_null_depth")

    # Curtosis de la parte difusa por tap (taps a menos de 30 dB del pico del PDP, sin el tap del pico)
    strong = real["pdp"] > real["pdp"].max() * 1e-3
    strong[np.argmax(real["pdp"])] = False  # el tap del pico va en la tabla (valores atípicos)
    strong = np.where(strong)[0]
    fig, ax = plt.subplots(figsize=(9, 5))
    for n, st in fakes.items():
        ax.plot(delay_us[strong], st["kurt_tap"][strong], color=COLORS[n], ls=LINESTYLES.get(n, "--"), lw=1.3,
                marker=".", ms=4, label=MODEL_LABELS[n])
    ax.plot(delay_us[strong], real["kurt_tap"][strong], color="black", lw=2, marker="o", ms=4, label="Real")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    _style(ax, r"Delay $\tau$ ($\mu$s)", "Excess kurtosis (diffuse part)")
    _save(fig, save_dir, "fig_kurtosis")

    # CDF de los parámetros por canal
    cp = real.get("channel_params", CHANNEL_PARAM_DEFAULTS)
    xl = {"ripple_db": r"Ripple $\max_f|H|^2/\min_f|H|^2$ (dB)",
          "fade_frac": f"Fraction of band in fade (< -{cp['fade_threshold_db']:g} dB)",
          "se": f"Spectral efficiency @ {cp['se_snr_db']:g} dB (b/s/Hz)",
          "bc_inst_mhz": f"Per-channel $B_c$ ({cp['bc_threshold']:g}) (MHz)",
          "delay_window_ns": f"Delay window {cp['delay_window_pct']:g}% (ns)",
          "n_taps_energy": f"Taps with {cp['delay_window_pct']:g}% of energy"}
    for thr in real.get("delay_thresholds", [None]):
        if thr is not None:
            xl[f"delay_interval{threshold_tag(thr)}"] = f"Delay interval @ {thr:g} dB (ns)"
    ncol = 4
    nrow = int(np.ceil(len(xl) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.8 * nrow), squeeze=False)
    for ax, (key, label) in zip(axes.ravel(), xl.items()):
        for n, st in fakes.items():
            v = np.sort(st[f"{key}_all"])
            ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=COLORS[n], ls=LINESTYLES.get(n, "--"), lw=1.2,
                    label=MODEL_LABELS[n])
        v = np.sort(real[f"{key}_all"])
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), color="black", lw=1.8, label="Real")
        allv = np.concatenate([real[f"{key}_all"]] + [st[f"{key}_all"] for st in fakes.values()])
        lo, hi = np.percentile(allv, [0.5, 99.5])
        ax.set_xlim(lo - 0.02 * (hi - lo), hi + 0.02 * (hi - lo))  # sin las colas extremas
        _style(ax, label, "CDF", legend=False)
        ax.xaxis.label.set_size(10)
    handles, labels_ = axes[0][0].get_legend_handles_labels()
    spare = axes.ravel()[len(xl):]
    for ax in spare:
        ax.axis("off")
    if len(spare):
        spare[0].legend(handles, labels_, loc="center", frameon=False, fontsize=11)
    else:
        axes[0][0].legend(fontsize=8, loc="lower right")
    _save(fig, save_dir, "fig_channel_params")

    # Distribución del RMS delay spread y del retardo medio, para cada umbral
    delay_thresholds = real.get("delay_thresholds", [None])
    delay_figs = []
    for thr in delay_thresholds:
        tag = threshold_tag(thr)
        suffix = "" if thr is None else f" (threshold {thr:g} dB)"
        delay_figs += [(f"rms_delay{tag}_all", "RMS delay spread" + suffix, f"fig_rms_delay{tag}"),
                       (f"mean_delay{tag}_all", "Mean delay" + suffix, f"fig_mean_delay{tag}")]
    for key, label, name in delay_figs:
        allv = np.concatenate([real[key]] + [st[key] for st in fakes.values()]) * 1e9
        xs = np.linspace(np.percentile(allv, 0.5), np.percentile(allv, 99.5), 400)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for n, st in fakes.items():
            axes[0].plot(xs, gaussian_kde(st[key] * 1e9)(xs), color=COLORS[n], ls=LINESTYLES.get(n, "--"),
                         lw=1.3,
                         label=MODEL_LABELS[n])
        axes[0].plot(xs, gaussian_kde(real[key] * 1e9)(xs), color="black", lw=2, label="Real")
        _style(axes[0], f"{label} (ns)", "Probability Density")
        names = ["Real"] + [MODEL_LABELS[n] for n in fakes]
        axes[1].boxplot([real[key] * 1e9] + [st[key] * 1e9 for st in fakes.values()],
                        tick_labels=names, showfliers=False)
        _style(axes[1], "", f"{label} (ns)", legend=False)
        _save(fig, save_dir, name)

    # CDF del RMS delay spread con todos los umbrales (una columna por umbral)
    nthr = len(delay_thresholds)
    fig, axes = plt.subplots(1, nthr, figsize=(4.2 * nthr, 4.2), squeeze=False)
    for ax, thr in zip(axes[0], delay_thresholds):
        key = f"rms_delay{threshold_tag(thr)}_all"
        for n, st in fakes.items():
            v = np.sort(st[key]) * 1e9
            ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=COLORS[n], ls=LINESTYLES.get(n, "--"), lw=1.3,
                    label=MODEL_LABELS[n])
        v = np.sort(real[key]) * 1e9
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), color="black", lw=2, label="Real")
        ax.set_title("No threshold" if thr is None else f"Threshold {thr:g} dB", fontsize=11)
        _style(ax, "RMS delay spread (ns)", "CDF" if ax is axes[0][0] else "",
               legend=ax is axes[0][0], loc="lower right")
    _save(fig, save_dir, "fig_rms_delay_cdf_thresholds")

    # Curvas de BER
    snr = np.array(ber_cfg["snr_db"], dtype=float)
    for mod, res in ber_results.items():
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.semilogy(snr, ber_theory_awgn(snr, mod), color="gray", lw=1, ls=":", label="AWGN (theory)")
        ax.semilogy(snr, ber_theory_rayleigh(snr, mod), color="gray", lw=1, ls="-.",
                    label="Rayleigh (theory)")
        for n, b in res.items():
            if n == "real" or n.startswith("_"):
                continue
            ax.semilogy(snr, np.where(b > 0, b, np.nan), color=COLORS[n], ls=LINESTYLES.get(n, "--"),
                        marker="x", ms=5,
                        lw=1.3, label=MODEL_LABELS[n])
        ax.semilogy(snr, np.where(res["real"] > 0, res["real"], np.nan), color="black", marker="o",
                    ms=5, lw=2, label="Real")
        ax.set_ylim(max(1e-6, 0.5 / res["_n_bits"]), 1.0)
        _style(ax, r"SNR $E_s/N_0$ (dB)", "BER")
        ax.grid(True, which="both", linestyle=":", alpha=0.6)
        _save(fig, save_dir, f"fig_ber_{mod.lower()}")

    # Ejemplos de canales (|h| en dB), real vs cada modelo
    real_ex, fake_ex = examples
    for n, fk in fake_ex.items():
        fig, axes = plt.subplots(2, 4, figsize=(16, 6), sharey=True)
        for i, ax in enumerate(axes.ravel()):
            src, lbl, col = (real_ex, "Real", "black") if i < 4 else (fk, MODEL_LABELS[n], COLORS[n])
            ax.plot(delay_us, 20 * np.log10(np.abs(src[i % 4]) + 1e-12), color=col, lw=1)
            ax.set_title(f"{lbl} #{i % 4 + 1}", fontsize=10)
            ax.grid(True, linestyle=":", alpha=0.6)
        for ax in axes[1]:
            ax.set_xlabel(r"Delay $\tau$ ($\mu$s)")
        for ax in axes[:, 0]:
            ax.set_ylabel(r"$|h[n]|$ (dB)")
        _save(fig, save_dir, f"examples_{n}")


# Tablas

def write_tables(save_dir, cfg_info, results, ber_results, ber_cfg, thresholds, delay_thresholds=(None,),
                 cp_cfg=None):
    models = list(results.keys())
    rows = [
        ("PDP MAE (lin)", "pdp_mae", ".3e"), ("PDP MSE (lin)", "pdp_mse", ".3e"),
        ("PDP MAE (dB)", "pdp_mae_db", ".3f"),
        ("Energy fake/real (dB)", "energy_ratio_db", ".3f"),
        ("Mean delay real (ns)", "mean_delay_real_ns", ".2f"),
        ("Mean delay fake (ns)", "mean_delay_fake_ns", ".2f"),
        ("Mean delay MAE (ns)", "mean_delay_mae_ns", ".2f"),
        ("Mean delay KS", "mean_delay_ks", ".3f"),
        ("Mean delay t-test p", "mean_delay_ttest_p", ".2e"),
        ("RMS delay real (ns)", "rms_delay_real_ns", ".2f"),
        ("RMS delay fake (ns)", "rms_delay_fake_ns", ".2f"),
        ("RMS delay MAE (ns)", "rms_delay_mae_ns", ".2f"),
        ("RMS delay KS", "rms_delay_ks", ".3f"),
        ("RMS delay t-test p", "rms_delay_ttest_p", ".2e"),
    ]
    for thr in delay_thresholds:
        if thr is None:
            continue
        tag, t = threshold_tag(thr), f"@{thr:g}dB"
        rows += [(f"Mean delay{t} real (ns)", f"mean_delay{tag}_real_ns", ".2f"),
                 (f"Mean delay{t} fake (ns)", f"mean_delay{tag}_fake_ns", ".2f"),
                 (f"Mean delay{t} KS", f"mean_delay{tag}_ks", ".3f"),
                 (f"RMS delay{t} real (ns)", f"rms_delay{tag}_real_ns", ".2f"),
                 (f"RMS delay{t} fake (ns)", f"rms_delay{tag}_fake_ns", ".2f"),
                 (f"RMS delay{t} MAE (ns)", f"rms_delay{tag}_mae_ns", ".2f"),
                 (f"RMS delay{t} KS", f"rms_delay{tag}_ks", ".3f")]
    rows += [
        ("Std per tap MAE", "std_tap_mae", ".3e"),
        ("|h| KS distance", "mag_ks", ".3f"),
        ("E|H(f)|^2 MAE (dB)", "freq_power_mae_db", ".3f"),
        ("Freq. corr. MAE", "freq_corr_mae", ".4f"),
        ("Null depth real (dB)", "null_depth_db_real", ".2f"),
        ("Null depth fake (dB)", "null_depth_db_fake", ".2f"),
        ("Null depth KS", "null_depth_db_ks", ".3f"),
        ("ZF noise gain real (dB)", "zf_gain_db_real", ".2f"),
        ("ZF noise gain fake (dB)", "zf_gain_db_fake", ".2f"),
        ("ZF noise gain KS", "zf_gain_db_ks", ".3f"),
        ("Kurtosis peak tap real", "kurt_peak_real", ".2f"),
        ("Kurtosis peak tap fake", "kurt_peak_fake", ".2f"),
        ("Kurtosis real (median)", "kurt_real", ".3f"),
        ("Kurtosis fake (median)", "kurt_fake", ".3f"),
        ("Kurtosis MAE", "kurt_mae", ".3f"),
    ]
    cp = cp_cfg or CHANNEL_PARAM_DEFAULTS
    labels = {"ripple_db": ("Ripple |H(f)| (dB)", ".2f"),
              "fade_frac": (f"Fade frac. (-{cp['fade_threshold_db']:g} dB)", ".4f"),
              "se": (f"SE @{cp['se_snr_db']:g}dB (b/s/Hz)", ".3f"),
              "bc_inst_mhz": (f"Bc{int(cp['bc_threshold'] * 100)} per ch. (MHz)", ".4f"),
              "delay_window_ns": (f"Delay window {cp['delay_window_pct']:g}% (ns)", ".1f"),
              "n_taps_energy": (f"Taps {cp['delay_window_pct']:g}% energy", ".1f")}
    for thr in delay_thresholds:
        if thr is not None:
            labels[f"delay_interval{threshold_tag(thr)}"] = (f"Delay interval@{thr:g}dB (ns)", ".1f")
    for key, (lbl, fmt) in labels.items():
        rows += [(f"{lbl} real", f"{key}_real", fmt), (f"{lbl} fake", f"{key}_fake", fmt),
                 (f"{lbl} KS", f"{key}_ks", ".3f")]
        if key == "se":
            rows += [(f"{lbl} out10% real", "se_out10_real", fmt),
                     (f"{lbl} out10% fake", "se_out10_fake", fmt)]
    rows += [
    ]
    for thr in thresholds:
        t = int(thr * 100)
        rows += [(f"Bc{t} real (MHz)", f"bc{t}_real_mhz", ".4f"),
                 (f"Bc{t} fake (MHz)", f"bc{t}_fake_mhz", ".4f"),
                 (f"Bc{t} error (MHz)", f"bc{t}_err_mhz", ".4f")]

    for key in sorted({k for r in results.values() for k in r if k.startswith("ber_")}):
        mod = key.split("_")[1]
        rows.append((f"BER {mod} |dlog10|", key, ".3f"))

    w = 34
    lines = ["===== EVALUACIÓN DE MODELOS GENERATIVOS =====", *cfg_info, ""]
    lines.append(f"{'Metric':<{w}}" + "".join(f"{MODEL_LABELS[m]:>12}" for m in models))
    lines.append("-" * (w + 12 * len(models)))
    for label, key, fmt in rows:
        lines.append(f"{label:<{w}}" + "".join(f"{results[m][key]:>12{fmt}}" for m in models))
    lines += ["", "Retardos: 'sin sufijo' = todos los taps; '@X dB' = solo taps con potencia >= pico del canal - X dB.",
              "Null depth / ZF noise gain: mediana por canal de min_f |H(f)|^2 y E_f{1/|H(f)|^2} (|H|^2 normalizado",
              "  a su media en la banda). Kurtosis: curtosis en exceso de la parte difusa (h - media del tap), media",
              "  de Re e Im (0 = gaussiana); 'peak tap' = tap del pico del PDP; median/MAE = resto de taps a menos",
              "  de 30 dB del pico.",
              "Parámetros por canal (mediana real/generada y KS): rizado max/min de |H(f)|^2; fracción de banda en",
              "  fade; eficiencia espectral E_f{log2(1+SNR|H|^2)} y su percentil 10 (outage 10%); Bc por canal (DFT",
              "  del perfil del propio canal); ventana e intervalo de retardo (ITU-R P.1407); taps con X% energía.",
              "KS: estadístico de Kolmogorov-Smirnov (0 = distribuciones idénticas, 1 = disjuntas).",
              "t-test p > 0.05: no hay diferencia significativa entre las medias."]
    txt = "\n".join(lines)
    with open(os.path.join(save_dir, "metrics_summary.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n" + txt)

    if ber_results:
        snr = ber_cfg["snr_db"]
        lines = []
        for mod, res in ber_results.items():
            names = ["real"] + [m for m in res if m not in ("real", "_n_bits")]
            lines.append(f"===== BER {mod} (ZF, {res['_n_bits']} bits por punto) =====")
            lines.append(f"{'SNR (dB)':<10}" + "".join(f"{(MODEL_LABELS.get(n, 'Real')):>12}" for n in names)
                         + f"{'AWGN th.':>12}{'Rayl. th.':>12}")
            for i, s in enumerate(snr):
                lines.append(f"{s:<10g}" + "".join(f"{res[n][i]:>12.3e}" for n in names)
                             + f"{ber_theory_awgn(s, mod):>12.3e}{ber_theory_rayleigh(s, mod):>12.3e}")
            lines.append("")
        txt = "\n".join(lines)
        with open(os.path.join(save_dir, "ber_table.txt"), "w", encoding="utf-8") as f:
            f.write(txt + "\n")
        print("\n" + txt)


# Programa principal

def main():
    parser = argparse.ArgumentParser(description="Evaluación de modelos generativos de canal")
    parser.add_argument("--config", default=os.path.join(THIS_DIR, "evaluation_config.json"))
    parser.add_argument("--models", nargs="+", choices=list(MODEL_LABELS), default=None,
                        help="Modelos a evaluar (por defecto los de la configuración)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--reuse-generated", action="store_true",
                        help="Usa los canales ya generados en save_dir/generated si existen")
    parser.add_argument("--no-ber", action="store_true", help="No calcula las curvas de BER")
    parser.add_argument("--subset", choices=["train", "val", "test", "all"], default=None,
                        help="Conjunto real de referencia si hay data.split_file (por defecto data.subset)")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config["experiment"]["seed"]
    set_seed(seed)
    ev = config["evaluation"]
    models = args.models or ev["models"]
    num_samples = ev["num_samples"]
    thresholds = ev.get("coherence_thresholds", [0.9, 0.5])
    # Umbrales del retardo medio / RMS delay spread (dB bajo el pico de cada canal); null = sin umbral
    channel_params = {**CHANNEL_PARAM_DEFAULTS, **ev.get("channel_params", {})}
    delay_thresholds = ev.get("delay_thresholds_db", [None, 40, 30, 25, 20])
    if None not in delay_thresholds:
        delay_thresholds = [None] + list(delay_thresholds)
    ber_cfg = config.get("ber", {})
    do_ber = ber_cfg.get("enabled", True) and not args.no_ber

    save_dir = repo_path(config["paths"]["save_dir"])
    gen_dir = os.path.join(save_dir, "generated")
    os.makedirs(gen_dir, exist_ok=True)

    dt = sampling_period(config)
    print(f"Periodo de muestreo de los taps: {dt * 1e9:.3f} ns (fs = {1 / dt / 1e6:.4f} MHz)")

    # Datos reales
    all_data = np.load(repo_path(config["data"]["path"])).astype(np.float32)
    real_data = all_data
    subset = args.subset or config["data"].get("subset", "all")
    if subset != "all":
        split_file = config["data"].get("split_file")
        if not split_file:
            raise ValueError("data.subset requiere data.split_file en la configuración")
        real_data = real_data[np.load(repo_path(split_file))[subset]]
    print(f"Referencia real: subconjunto '{subset}' ({len(real_data)} canales)")
    # Datos con los que se ajustan los modelos clásicos (baselines.py): train si hay separación
    split_file = config["data"].get("split_file")
    train_data = all_data[np.load(repo_path(split_file))["train"]] if split_file else all_data
    N_real, _, L = real_data.shape
    assert L == config["data"]["signal_length"], "signal_length no coincide con el dataset"
    h_real = to_complex(real_data)
    rng = np.random.RandomState(seed)
    real_subset = h_real[rng.choice(N_real, min(num_samples, N_real), replace=False)]
    print(f"Dataset real: {real_data.shape} (BER con un subconjunto de {len(real_subset)})")

    # Generar o cargar canales
    fake = {}
    for mt in models:
        out_path = os.path.join(gen_dir, f"h_generated_{mt}.npy")
        mcfg = config["models"].get(mt, {})
        ckpt = repo_path(mcfg["checkpoint"]) if "checkpoint" in mcfg else ""
        if mt in BASELINES:
            # Modelo clásico: se ajusta con train (es rápido, siempre se regenera)
            print(f"[{MODEL_LABELS[mt]}] ajustando con train ({len(train_data)} canales), "
                  f"generando {num_samples}")
            data = generate_baseline(mt, train_data, num_samples, seed)
            np.save(out_path, data)
        elif args.reuse_generated and os.path.exists(out_path):
            print(f"[{MODEL_LABELS[mt]}] reutilizando {out_path}")
            data = np.load(out_path)
        elif os.path.exists(ckpt):
            print(f"[{MODEL_LABELS[mt]}] generando {num_samples} canales con {ckpt} ...")
            data = generate_signals(mt, {**mcfg, "checkpoint": ckpt}, num_samples, L, args.device,
                                    real_data, seed, batch_size=ev.get("batch_size", 500))
            np.save(out_path, data)
        elif os.path.exists(out_path):
            print(f"[{MODEL_LABELS[mt]}] no existe el checkpoint, uso los canales guardados en {out_path}")
            data = np.load(out_path)
        else:
            print(f"[{MODEL_LABELS[mt]}] AVISO: no hay checkpoint ({ckpt}) ni canales generados; se omite")
            continue
        if data.shape[1:] != (2, L):
            raise ValueError(f"{mt}: forma {data.shape}, se esperaba (N, 2, {L})")
        fake[mt] = to_complex(data)

    if not fake:
        print("No hay ningún modelo que evaluar.")
        return

    # Rejillas comunes para CDF/KDE de |h|
    max_mag = max(np.abs(h_real).max(), *(np.abs(f).max() for f in fake.values()))
    mag_grid = np.linspace(0, max_mag, 500)
    mag_db_grid = np.linspace(-80, 20 * np.log10(max_mag + 1e-12), 500)

    print("\nCalculando métricas ...")
    kw = dict(seed=seed, delay_thresholds=delay_thresholds, channel_params=channel_params)
    real_st = channel_stats(h_real, dt, mag_grid, mag_db_grid, **kw)
    fake_st = {mt: channel_stats(h, dt, mag_grid, mag_db_grid, **kw) for mt, h in fake.items()}
    df = 1 / dt / L
    results = {mt: compare(real_st, st, df, thresholds) for mt, st in fake_st.items()}

    # BER (mismos bits y ruido para todos los conjuntos de canales)
    ber_results = {}
    if do_ber:
        snr = ber_cfg.get("snr_db", list(range(0, 31, 2)))
        ber_cfg["snr_db"] = snr
        for mod in ber_cfg.get("modulations", ["QPSK"]):
            print(f"BER {mod} ...")
            kw = dict(mod_type=mod, nfft=ber_cfg.get("nfft"), n_ofdm=ber_cfg.get("n_ofdm", 1), seed=seed)
            res = {"real": ber_curve(real_subset, snr, **kw)}
            for mt, h in fake.items():
                res[mt] = ber_curve(h[: len(real_subset)], snr, **kw)
            nfft = ber_cfg.get("nfft") or L
            res["_n_bits"] = len(real_subset) * nfft * ber_cfg.get("n_ofdm", 1) * (2 if mod == "QPSK" else 4)
            ber_results[mod] = res
            for mt in fake:
                results[mt][f"ber_{mod}_mae_log10"] = float(np.nanmean(np.abs(
                    np.log10(np.maximum(res[mt], 1e-12)) - np.log10(np.maximum(res["real"], 1e-12)))))

    # Salidas
    cfg_info = [f"Datos reales: {config['data']['path']}, subconjunto '{subset}' ({N_real} canales, L={L})",
                f"Muestras generadas por modelo: {num_samples}",
                f"Periodo de muestreo: {dt * 1e9:.3f} ns"]
    write_tables(save_dir, cfg_info, results, ber_results, ber_cfg, thresholds, delay_thresholds,
                 channel_params)

    ex_rng = np.random.default_rng(seed)
    examples = (h_real[ex_rng.choice(N_real, 4, replace=False)], {mt: h[:4] for mt, h in fake.items()})
    plot_all(save_dir, real_st, fake_st, (mag_grid, mag_db_grid), dt, thresholds, ber_results, ber_cfg,
             examples)

    strip = lambda d: {k: v for k, v in d.items() if not k.startswith("_")}  # noqa: E731
    with open(os.path.join(save_dir, "metrics.pkl"), "wb") as f:
        pickle.dump({"real": strip(real_st), "fake": {k: strip(v) for k, v in fake_st.items()},
                     "metrics": results, "ber": ber_results, "grids": (mag_grid, mag_db_grid),
                     "dt": dt, "config": config}, f)
    print(f"\nResultados guardados en {save_dir}")


if __name__ == "__main__":
    main()
