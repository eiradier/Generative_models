"""
Evaluación de los modelos generativos de canal (GAN, DCGAN, WGAN, DDPM, DDIM) frente a
las medidas reales de loopback.

Para cada modelo:
  1. Genera `num_samples` canales con el checkpoint entrenado (o reutiliza los ya generados).
  2. Calcula métricas frente al dataset real: PDP, retardo medio y RMS delay spread, distribución
     de |h|, potencia media de la respuesta en frecuencia, función de correlación en frecuencia,
     ancho de banda de coherencia y curvas de BER (QPSK/16QAM con ZF).
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
from scipy.stats import gaussian_kde, ks_2samp, ttest_ind

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, REPO_ROOT)

from BER import ber_curve, ber_theory_awgn, ber_theory_rayleigh  # noqa: E402
from train_models.normalization import TapScaler  # noqa: E402

MODEL_LABELS = {"gan": "GAN", "dcgan": "DCGAN", "wgan": "WGAN", "ddpm": "DDPM", "ddim": "DDIM"}
COLORS = {"gan": "red", "dcgan": "blue", "wgan": "purple", "ddpm": "green", "ddim": "orange"}

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
        for i in range(0, num_samples, batch_size):
            b = min(batch_size, num_samples - i)
            x = torch.randn(b, 2, L, device=device, generator=g)
            for t in reversed(range(ddpm.T)):
                t_batch = torch.full((b,), t, device=device, dtype=torch.long)
                eps = ddpm.model(x, t_batch)
                beta = ddpm.betas[t]
                x = (x - beta / torch.sqrt(1 - ddpm.alphas_cumprod[t]) * eps) / torch.sqrt(1 - beta)
                if t > 0:
                    x = x + torch.sqrt(beta) * torch.randn(x.shape, device=device, generator=g)
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

def channel_stats(h, dt, mag_grid, mag_db_grid, kde_points=50000, seed=0):
    """Estadísticas de un conjunto de canales complejos h (N, L)."""
    N, L = h.shape
    p = np.abs(h) ** 2
    energy = p.sum(axis=1)
    tau = np.arange(L) * dt

    mean_delay = (p * tau).sum(axis=1) / (energy + 1e-30)
    rms_delay = np.sqrt((p * (tau[None] - mean_delay[:, None]) ** 2).sum(axis=1) / (energy + 1e-30))

    pdp = p.mean(axis=0)

    # Respuesta en frecuencia: potencia media por subportadora
    H = np.fft.fft(h, axis=1)
    freq_power = np.fft.fftshift((np.abs(H) ** 2).mean(axis=0))

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
        "freq_corr": freq_corr,
        "mag_cdf": cdf, "mag_cdf_db": cdf_db,
        "mag_kde": kde, "mag_kde_db": kde_db,
        "_mag_sub": mag[sub],
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

    for k in ("mean_delay", "rms_delay"):
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
    for thr in thresholds:
        bc_r = coherence_bandwidth(real["freq_corr"], df, thr)
        bc_f = coherence_bandwidth(fake["freq_corr"], df, thr)
        m[f"bc{int(thr * 100)}_real_mhz"] = bc_r / 1e6
        m[f"bc{int(thr * 100)}_fake_mhz"] = bc_f / 1e6
        m[f"bc{int(thr * 100)}_err_mhz"] = abs(bc_r - bc_f) / 1e6
    return m


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
        ax.plot(x, transform(st[key]), label=MODEL_LABELS[name], color=COLORS[name], lw=1.3, ls="--")
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
        axins.plot(x, st["mag_cdf"], color=COLORS[name], lw=1.1, ls="--")
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

    # Distribución del RMS delay spread y del retardo medio
    for key, label, name in (("rms_delay_all", "RMS delay spread", "fig_rms_delay"),
                             ("mean_delay_all", "Mean delay", "fig_mean_delay")):
        allv = np.concatenate([real[key]] + [st[key] for st in fakes.values()]) * 1e9
        xs = np.linspace(np.percentile(allv, 0.5), np.percentile(allv, 99.5), 400)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for n, st in fakes.items():
            axes[0].plot(xs, gaussian_kde(st[key] * 1e9)(xs), color=COLORS[n], ls="--", lw=1.3,
                         label=MODEL_LABELS[n])
        axes[0].plot(xs, gaussian_kde(real[key] * 1e9)(xs), color="black", lw=2, label="Real")
        _style(axes[0], f"{label} (ns)", "Probability Density")
        names = ["Real"] + [MODEL_LABELS[n] for n in fakes]
        axes[1].boxplot([real[key] * 1e9] + [st[key] * 1e9 for st in fakes.values()],
                        tick_labels=names, showfliers=False)
        _style(axes[1], "", f"{label} (ns)", legend=False)
        _save(fig, save_dir, name)

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
            ax.semilogy(snr, np.where(b > 0, b, np.nan), color=COLORS[n], ls="--", marker="x", ms=5,
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

def write_tables(save_dir, cfg_info, results, ber_results, ber_cfg, thresholds):
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
        ("Std per tap MAE", "std_tap_mae", ".3e"),
        ("|h| KS distance", "mag_ks", ".3f"),
        ("E|H(f)|^2 MAE (dB)", "freq_power_mae_db", ".3f"),
        ("Freq. corr. MAE", "freq_corr_mae", ".4f"),
    ]
    for thr in thresholds:
        t = int(thr * 100)
        rows += [(f"Bc{t} real (MHz)", f"bc{t}_real_mhz", ".4f"),
                 (f"Bc{t} fake (MHz)", f"bc{t}_fake_mhz", ".4f"),
                 (f"Bc{t} error (MHz)", f"bc{t}_err_mhz", ".4f")]

    for key in sorted({k for r in results.values() for k in r if k.startswith("ber_")}):
        mod = key.split("_")[1]
        rows.append((f"BER {mod} |dlog10|", key, ".3f"))

    w = 24
    lines = ["===== EVALUACIÓN DE MODELOS GENERATIVOS =====", *cfg_info, ""]
    lines.append(f"{'Metric':<{w}}" + "".join(f"{MODEL_LABELS[m]:>12}" for m in models))
    lines.append("-" * (w + 12 * len(models)))
    for label, key, fmt in rows:
        lines.append(f"{label:<{w}}" + "".join(f"{results[m][key]:>12{fmt}}" for m in models))
    lines += ["", "KS: estadístico de Kolmogorov-Smirnov (0 = distribuciones idénticas, 1 = disjuntas).",
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
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config["experiment"]["seed"]
    set_seed(seed)
    ev = config["evaluation"]
    models = args.models or ev["models"]
    num_samples = ev["num_samples"]
    thresholds = ev.get("coherence_thresholds", [0.9, 0.5])
    ber_cfg = config.get("ber", {})
    do_ber = ber_cfg.get("enabled", True) and not args.no_ber

    save_dir = repo_path(config["paths"]["save_dir"])
    gen_dir = os.path.join(save_dir, "generated")
    os.makedirs(gen_dir, exist_ok=True)

    dt = sampling_period(config)
    print(f"Periodo de muestreo de los taps: {dt * 1e9:.3f} ns (fs = {1 / dt / 1e6:.4f} MHz)")

    # Datos reales
    real_data = np.load(repo_path(config["data"]["path"])).astype(np.float32)
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
        mcfg = config["models"][mt]
        ckpt = repo_path(mcfg["checkpoint"])
        if args.reuse_generated and os.path.exists(out_path):
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
    real_st = channel_stats(h_real, dt, mag_grid, mag_db_grid, seed=seed)
    fake_st = {mt: channel_stats(h, dt, mag_grid, mag_db_grid, seed=seed) for mt, h in fake.items()}
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
    cfg_info = [f"Datos reales: {config['data']['path']} ({N_real} canales, L={L})",
                f"Muestras generadas por modelo: {num_samples}",
                f"Periodo de muestreo: {dt * 1e9:.3f} ns"]
    write_tables(save_dir, cfg_info, results, ber_results, ber_cfg, thresholds)

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
