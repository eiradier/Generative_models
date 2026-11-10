import argparse
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.stats import gaussian_kde

import json
import importlib

from BER import compute_ber

def load_config(path="config/evaluation_config.json"):
    with open(path, "r") as f:
        return json.load(f)

def to_db(x, eps=1e-10):
    return 10 * np.log10(np.maximum(x, eps))

def mae(a, b):
    return np.mean(np.abs(a - b))

def load_model_bundle(model_type):
    module_map = {
        "gan": ("train_modelos.train_GAN_Conv1D", ["Generator"]),
        "dcgan": ("train_modelos.train_DCGAN_Conv1D", ["Generator"]),
        "wgan": ("train_modelos.train_WGAN", ["Generator"]),
        "ddpm": ("train_modelos.train_DDPM", ["UNet1D", "DDPM"]),
        "ddim": ("train_modelos.train_DDIM", ["UNet1D", "Diffusion", "DDIMSampler"]),
    }

    if model_type not in module_map:
        raise ValueError(f"Modelo no soportado: {model_type}")

    module_name, attrs = module_map[model_type]
    m = importlib.import_module(module_name)

    return {a: getattr(m, a) for a in attrs}

def compute_pdp(signals):
    power = np.abs(signals) ** 2
    return np.mean(power, axis=0)

def compute_energy(signals):
    """Energía total de cada señal"""
    return np.sum(np.abs(signals)**2, axis=1)

def compute_std_per_tap(signals):
    """Desviación estándar por tap (columnas)"""
    return np.std(signals, axis=0)

def compute_autocorrelation(signals):
    """Autocorrelación promedio por señal"""
    N, L = signals.shape
    autocorrelation = np.zeros(L)
    for s in signals:
        s = s - np.mean(s)
        ac = np.correlate(s, s, mode='full')
        ac = ac[ac.size//2:] # solo la mitad positiva
        autocorrelation += ac / (np.max(ac) + 1e-12)
    return autocorrelation / N

def compute_average_delay(signals, delta_tau=1.0):
    """
    Average delay por señal y luego promedio global.
    """
    N, L = signals.shape
    delays = np.arange(L) * delta_tau
    avg_delays = []
    for s in signals:
        power = np.abs(s) ** 2
        total_power = np.sum(power) + 1e-12
        mean_delay = np.sum(delays * power) / total_power
        avg_delays.append(mean_delay)

    avg_delays = np.array(avg_delays)
    return np.mean(avg_delays), avg_delays

def compute_rms_delay_spread(signals, delta_tau=1.0):
    """
    RMS delay spread por señal y luego promedio global.
    """
    N, L = signals.shape
    delays = np.arange(L) * delta_tau
    rms_values = []
    for s in signals:
        power = np.abs(s) ** 2
        total_power = np.sum(power) + 1e-12
        mean_delay = np.sum(delays * power) / total_power
        rms = np.sqrt(np.sum(power * (delays - mean_delay) ** 2) / total_power)
        rms_values.append(rms)

    rms_values = np.array(rms_values)
    return np.mean(rms_values), rms_values

def compute_histogram(signals, bins=50, range=(-1,1)):
    """Histograma normalizado de magnitudes |h[n]|"""
    mags = np.abs(signals).flatten()
    hist, edges = np.histogram(mags, bins=bins, range=range, density=True)
    return hist, edges[:-1]

def compute_psd(signals):
    """PSD promedio de las señales"""
    psd = np.abs(np.fft.rfft(signals, axis=-1)) ** 2
    psd_mean = psd.mean(axis=0)

    freqs = np.fft.rfftfreq(signals.shape[-1], d=1.0)
    return psd_mean, freqs

def evaluate_metrics(real, fake, delta_tau=1.0):
    m = {}

    # PDP
    m["pdp_real"] = compute_pdp(real)
    m["pdp_fake"] = compute_pdp(fake)

    m["pdp_mae"] = mae(m["pdp_real"], m["pdp_fake"])

    m["pdp_real_db"] = to_db(m["pdp_real"])
    m["pdp_fake_db"] = to_db(m["pdp_fake"])

    mask = m["pdp_real"] > 1e-6
    m["pdp_mae_db"] = np.mean(
        np.abs(m["pdp_real_db"][mask] - m["pdp_fake_db"][mask])
    )

    # Delay
    m["avg_real"], m["avg_real_all"] = compute_average_delay(real, delta_tau)
    m["avg_fake"], m["avg_fake_all"] = compute_average_delay(fake, delta_tau)

    m["avg_delay_mae"] = abs(m["avg_real"] - m["avg_fake"])
    m["avg_delay_mae_ns"] = m["avg_delay_mae"] * 1e9

    m["rms_real"], m["rms_real_all"] = compute_rms_delay_spread(real, delta_tau)
    m["rms_fake"], m["rms_fake_all"] = compute_rms_delay_spread(fake, delta_tau)

    m["rms_mae"] = abs(m["rms_real"] - m["rms_fake"])
    m["rms_mae_ns"] = m["rms_mae"] * 1e9

    # tests
    m["rms_t_stat"], m["rms_t_pvalue"] = ttest_ind(
        m["rms_real_all"], m["rms_fake_all"], equal_var=False
    )

    m["avg_t_stat"], m["avg_t_pvalue"] = ttest_ind(
        m["avg_real_all"], m["avg_fake_all"], equal_var=False
    )

    # STD
    m["std_real"] = compute_std_per_tap(real)
    m["std_fake"] = compute_std_per_tap(fake)
    m["std_mae"] = mae(m["std_real"], m["std_fake"])

    # autocorr
    m["autocorr_real"] = compute_autocorrelation(real)
    m["autocorr_fake"] = compute_autocorrelation(fake)
    m["autocorr_mae"] = mae(m["autocorr_real"], m["autocorr_fake"])

    # hist
    m["hist_real"], m["bins_real"] = compute_histogram(real)
    m["hist_fake"], m["bins_fake"] = compute_histogram(fake)

    # PSD
    m["psd_real"], m["freqs"] = compute_psd(real)
    m["psd_fake"], _ = compute_psd(fake)

    m["psd_mae"] = mae(m["psd_real"], m["psd_fake"])

    m["psd_real_db"] = to_db(m["psd_real"])
    m["psd_fake_db"] = to_db(m["psd_fake"])
    m["psd_mae_db"] = mae(m["psd_real_db"], m["psd_fake_db"])

    return m

def generate_signals(config, model_type, model_checkpoint, num_samples,
                     device, model_bundle, seed):
    """
    Genera señales sintéticas de canal utilizando un modelo ya entrenado,
    sin alterar los pesos ni la distribución aprendida.
    Para obtener señales falsas comparables con las reales.
    Return:
    array (num_samples, L) con las señales generadas (normalizadas en [-1,1]).
    """
    L = config["data"]["signal_length"]
    z_dim = config["models"][model_type].get("z_dim", None)
    dm_steps = config["models"][model_type].get("T", 1000)
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    if model_type == 'gan':
        checkpoint = torch.load(model_checkpoint, map_location=device)
        Generator = model_bundle["Generator"]
        G = Generator(z_dim=z_dim, L=L).to(device)
        G.load_state_dict(checkpoint["G"])
        G.eval()

        with torch.no_grad():
            z = torch.randn(num_samples, z_dim, device=device, generator=g)
            fake = G(z).cpu().numpy()

    # DCGAN: Generación directa a partir de ruido latente z
    elif model_type == 'dcgan':
        # cargar checkpoint y reconstruir el generador
        checkpoint = torch.load(model_checkpoint, map_location=device)
        Generator = model_bundle["Generator"]
        G = Generator(z_dim=z_dim, L=L).to(device)
        G.load_state_dict(checkpoint['G'])
        G.eval()

        # Muestreo directo: x = G(z)
        with torch.no_grad():
            z = torch.randn(num_samples, z_dim, device=device, generator=g)
            fake = G(z).squeeze(1).cpu().numpy()

    # WGAN-GP
    elif model_type == 'wgan':
        checkpoint = torch.load(model_checkpoint, map_location=device)
        Generator = model_bundle["Generator"]
        G = Generator(z_dim=z_dim, L=L).to(device)
        G.load_state_dict(checkpoint['G'])
        G.eval()

        with torch.no_grad():
            z = torch.randn(num_samples, z_dim, device=device, generator=g)
            fake = G(z).cpu().numpy()

    # DDPM: generación mediante proceso inverso de difusión
    elif model_type == 'ddpm':
        # cargar U-Net entrenada
        UNet1D = model_bundle["UNet1D"]
        DDPM = model_bundle["DDPM"]
        checkpoint = torch.load(model_checkpoint, map_location=device)
        unet = UNet1D(time_emb_dim=config["models"]["ddpm"]["time_emb_dim"]).to(device)
        unet.load_state_dict(checkpoint)

        # reconstruir DDPM con el mismo número de pasos T
        ddpm = DDPM(unet, T=dm_steps).to(device)
        ddpm.model.eval()

        with torch.no_grad():
            # Inicialización desde ruido gaussiano puro
            x = torch.randn(num_samples, 2, L, device=device, generator=g)

            # Proceso inverso de difusión: eliminar ruido paso a paso
            # desde t=T-1 hasta t=0
            for t in reversed(range(ddpm.T)):
                t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
                noise_pred = ddpm.model(x, t_batch)

                beta = ddpm.betas[t]
                if t > 0:
                    z = torch.randn_like(x, generator=g)
                else:
                    z = torch.zeros_like(x)

                # ecuación de muestreo DDPM (simplificada)
                x = (
                        (1 / torch.sqrt(1 - beta)) *
                        (x - beta / torch.sqrt(1 - ddpm.alphas_cumprod[t]) * noise_pred)
                        + torch.sqrt(beta) * z
                )
            # guardar señal final x_0
            x0 = x
            x0 = torch.clamp(x0, -1, 1)
            fake = x0.cpu().numpy()

    elif model_type == 'ddim':
        UNet1D = model_bundle["UNet1D"]
        Diffusion = model_bundle["Diffusion"]
        DDIMSampler = model_bundle["DDIMSampler"]
        checkpoint = torch.load(model_checkpoint, map_location=device)

        unet = UNet1D(
            time_emb_dim=config["models"]["ddim"]["time_emb_dim"]
        ).to(device)

        unet.load_state_dict(checkpoint)
        unet.eval()

        data = np.load(config["data"]["path"]).astype(np.float32)
        diffusion = Diffusion(unet, T=dm_steps, data=data).to(device)

        sampler = DDIMSampler(
            diffusion,
            eta=config["models"]["ddim"]["eta"],
        )

        with torch.no_grad():
            return sampler.sample(
                shape=(num_samples, 2, L),
                device=device,
                steps=config["models"]["ddim"]["sampling_steps"]
            ).cpu().numpy()
    else:
        raise ValueError(model_type)

    return fake

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_generated_dataset(fake, args, config):
    out_dir = os.path.join(args.save_dir, "dataset_generado")
    os.makedirs(out_dir, exist_ok=True)

    np.save(
        os.path.join(out_dir, "h_generated.npy"),
        fake.astype(np.float32)
    )

    with open(os.path.join(out_dir, "info.txt"), "w") as f:
        f.write(f"model_type: {args.model_type}\n")
        f.write(f"checkpoint: {args.model}\n")
        f.write(f"samples: {args.num_eval}\n")
        f.write(f"signal_length: {fake.shape[-1]}\n")

def main(args, config):
    seed = config["experiment"]["seed"]
    set_seed(seed)

    print("\n=== Cargando datos ===")
    real_eval = load_real_data(args.data, args.num_eval, seed)
    print("Dataset cargado:", real_eval.shape)

    print(f"\n=== Generando señales con {args.model_type.upper()} ===")
    model_bundle = load_model_bundle(args.model_type)

    fake_eval = generate_signals(
        config=config,
        model_type=args.model_type,
        model_checkpoint=args.model,
        num_samples=args.num_eval,
        device=args.device,
        model_bundle=model_bundle,
        seed=seed
    )

    print("Señales generadas:", fake_eval.shape)
    save_generated_dataset(fake_eval, args, config)

    # convertir a canal complejo
    h_real = real_eval[:, 0, :] + 1j*real_eval[:, 1, :]
    h_fake = fake_eval[:, 0, :] + 1j*fake_eval[:, 1, :]

    N, L = h_real.shape
    num_symbols = N * L
    num_bits = num_symbols * 2  # QPSK

    bits = np.random.randint(0, 2, num_bits)

    # BER
    ber_real = compute_ber(h_real, bits, mod_type="QPSK")
    ber_fake = compute_ber(h_fake, bits, mod_type="QPSK")

    print("\n--- BER ---")
    print(f"BER real: {ber_real:.6f}")
    print(f"BER fake: {ber_fake:.6f}")
    print(f"BER diff: {abs(ber_real - ber_fake):.6f}")

    real_mag, fake_mag = to_magnitude(real_eval, fake_eval)

    metrics = evaluate_metrics(real_mag, fake_mag, delta_tau=args.delta_tau)
    print_metrics(metrics)
    print_physical_metrics(metrics)
    save_metrics(args, metrics)
    write_metrics_file(args, metrics, ber_real, ber_fake)
    plot_results(args, metrics, real_mag, fake_mag)

    return metrics

def load_real_data(path, num_eval, seed=42):
    data = np.load(path).astype(np.float32)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(data), num_eval, replace=False)
    return data[idx]

def to_magnitude(real_eval, fake_eval):
    real_complex = real_eval[:, 0, :] + 1j * real_eval[:, 1, :]
    fake_complex = fake_eval[:, 0, :] + 1j * fake_eval[:, 1, :]
    return np.abs(real_complex), np.abs(fake_complex)

def print_metrics(metrics):
    print("\n--- RESULTADOS ---")
    for key in [
        'pdp_mae', 'pdp_mae_db',
        'avg_delay_mae_ns', 'rms_mae_ns',
        'std_mae', 'autocorr_mae',
        'psd_mae', 'psd_mae_db'
    ]:
        val = metrics[key]

        if key in ['avg_delay_mae_ns', 'rms_mae_ns']:
            print(f"{key}: {val:.3f} ns")
        elif key == 'pdp_mae_db':
            print(f"{key}: {val:.3f} dB")
        elif key == 'psd_mae_db':
            print(f"{key}: {val:.3f} dB/Hz")
        else:
            print(f"{key}: {val:.6f}")

def print_physical_metrics(metrics):
    ns = 1e9

    print("\n--- MÉTRICAS REALES ---")
    print(f"Avg delay real: {metrics['avg_real'] * ns:.3f} ns")
    print(f"RMS delay real: {metrics['rms_real'] * ns:.3f} ns")
    print(f"Mean PSD real: {np.mean(metrics['psd_real_db']):.3f} dB/Hz")

    print("\n--- MÉTRICAS GENERADAS ---")
    print(f"Avg delay fake: {metrics['avg_fake'] * ns:.3f} ns")
    print(f"RMS delay fake: {metrics['rms_fake'] * ns:.3f} ns")
    print(f"Mean PSD fake: {np.mean(metrics['psd_fake_db']):.3f} dB/Hz")

def save_metrics(args, metrics):
    os.makedirs(args.save_dir, exist_ok=True)
    np.save(os.path.join(args.save_dir, "metrics.npy"), metrics)

def write_metrics_file(args, metrics, ber_real=None, ber_fake=None):
    scalar_keys = [
        'pdp_mae', 'pdp_mae_db',
        'avg_delay_mae_ns', 'rms_mae_ns',
        'std_mae', 'autocorr_mae',
        'psd_mae', 'psd_mae_db'
    ]

    def significance(p):
        return "STATISTICALLY SIMILAR" if p > 0.05 else "STATISTICALLY DIFFERENT"

    txt_path = os.path.join(args.save_dir, "metrics.txt")

    with open(txt_path, "w") as f:
        f.write("===== RESULTADOS EVALUACION =====\n")
        f.write(f"Model type: {args.model_type}\n")
        f.write(f"Checkpoint: {args.model}\n")
        f.write(f"Num eval samples: {args.num_eval}\n")
        f.write("=" * 50 + "\n\n")

        f.write("---- METRICAS GLOBALES ----\n")
        for k in scalar_keys:
            if k in ['avg_delay_mae_ns', 'rms_mae_ns']:
                f.write(f"{k}: {metrics[k]:.3f} ns\n")
            elif k == 'pdp_mae_db':
                f.write(f"{k}: {metrics[k]:.3f} dB\n")
            elif k == 'psd_mae_db':
                f.write(f"{k}: {metrics[k]:.3f} dB/Hz\n")
            else:
                f.write(f"{k}: {metrics[k]:.6f}\n")

        f.write("\n---- STATISTICAL TESTS ----\n")
        f.write(f"RMS t-test statistic: {metrics['rms_t_stat']:.6f}\n")
        f.write(f"RMS t-test p_value: {metrics['rms_t_pvalue']:.3e}\n")
        f.write(f"AVG delay t-test statistic: {metrics['avg_t_stat']:.6f}\n")
        f.write(f"AVG delay t-test p_value: {metrics['avg_t_pvalue']:.3e}\n")

        f.write("\n---- REAL CHANNEL STATS ----\n")
        f.write(f"Avg delay real: {metrics['avg_real'] * 1e9:.3f} ns\n")
        f.write(f"RMS delay real: {metrics['rms_real'] * 1e9:.3f} ns\n")
        f.write(f"Mean PSD real: {np.mean(metrics['psd_real_db']):.3f} dB/Hz\n")

        f.write("\n---- GENERATED CHANNEL STATS ----\n")
        f.write(f"Avg delay fake: {metrics['avg_fake'] * 1e9:.3f} ns\n")
        f.write(f"RMS delay fake: {metrics['rms_fake'] * 1e9:.3f} ns\n")
        f.write(f"Mean PSD fake: {np.mean(metrics['psd_fake_db']):.3f} dB/Hz\n")

        f.write("\n---- BER ----\n")
        f.write(f"BER real: {ber_real:.6f}\n")
        f.write(f"BER fake: {ber_fake:.6f}\n")
        f.write(f"BER diff: {abs(ber_real - ber_fake):.6f}\n")

        f.write("\n---- STATISTICAL INTERPRETATION ----\n")
        f.write(f"RMS comparison: {significance(metrics['rms_t_pvalue'])}\n")
        f.write(f"AVG delay comparison: {significance(metrics['avg_t_pvalue'])}\n")

def plot_delay_distribution(args, metrics):
    plt.figure(figsize=(8, 4))

    plt.hist(
        metrics["rms_real_all"] * 1e9,
        bins=30,
        alpha=0.6,
        density=True,
        label="Real"
    )

    plt.hist(
        metrics["rms_fake_all"] * 1e9,
        bins=30,
        alpha=0.6,
        density=True,
        label="Generado"
    )

    plt.xlabel("RMS Delay Spread (ns)")
    plt.ylabel("Density")
    plt.title("Distribución RMS Delay Spread")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(args.save_dir, "rms_distribution.png")
    )


def plot_boxplot_delay(args, metrics):
    plt.figure(figsize=(6, 4))

    plt.boxplot([
        metrics["rms_real_all"] * 1e9,
        metrics["rms_fake_all"] * 1e9
    ],
    tick_labels=["Real", "Generado"])

    plt.ylabel("RMS Delay Spread (ns)")
    plt.title("Boxplot RMS Delay Spread")
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(args.save_dir, "boxplot_rms.png")
    )


def plot_cdf(args, real_mag, fake_mag):
    real_vals = real_mag.flatten()
    fake_vals = fake_mag.flatten()

    real_sorted = np.sort(real_vals)
    fake_sorted = np.sort(fake_vals)

    real_cdf = np.arange(len(real_sorted)) / len(real_sorted)
    fake_cdf = np.arange(len(fake_sorted)) / len(fake_sorted)

    plt.figure(figsize=(8, 4))

    plt.plot(real_sorted, real_cdf, label="Real", linewidth=2)
    plt.plot(fake_sorted, fake_cdf, label="Generado", linewidth=2, linestyle='--')
    plt.grid(True, alpha=0.3)
    plt.xlabel("|h[n]|")
    plt.ylabel("CDF")
    plt.title("CDF de magnitudes")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(args.save_dir, "cdf_magnitudes.png")
    )


def plot_kde(args, real_mag, fake_mag):
    real_vals = real_mag.flatten()
    fake_vals = fake_mag.flatten()

    kde_real = gaussian_kde(real_vals)
    kde_fake = gaussian_kde(fake_vals)

    x = np.linspace(
        0,
        max(real_vals.max(), fake_vals.max()),
        500
    )

    plt.figure(figsize=(8, 4))

    plt.plot(x, kde_real(x), label="Real", linewidth=2)
    plt.plot(x, kde_fake(x), label="Generado", linewidth=2, linestyle='--')
    plt.grid(True, alpha=0.3)
    plt.xlabel("|h[n]|")
    plt.ylabel("Density")
    plt.title("Kernel Density Estimation")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(args.save_dir, "kde_distribution.png")
    )

def plot_results(args, metrics, real_mag, fake_mag):
    plt.figure(figsize=(8, 4))
    delays = np.arange(len(metrics['pdp_real_db'])) * args.delta_tau * 1e9
    stride = max(1, len(metrics['pdp_real_db']) // 15)

    plt.plot(
        delays,
        metrics['pdp_real_db'],
        label="Real",
        linewidth=2,
        marker='o',
        markersize=4,
        markevery=stride
    )

    plt.plot(
        delays,
        metrics['pdp_fake_db'],
        label="Generado",
        linewidth=2,
        linestyle='--',
        marker='x',
        markersize=4,
        markevery=stride
    )
    plt.grid(True, alpha=0.3)
    plt.title("PDP (dB) Real vs Generado")
    plt.xlabel("Delay (ns)")
    plt.ylabel("Power (dB)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "pdp.png"))

    plt.figure(figsize=(8, 4))
    plt.bar(metrics['bins_real'], metrics['hist_real'], width=0.03, alpha=0.6, label="Real")
    plt.bar(metrics['bins_fake'], metrics['hist_fake'], width=0.03, alpha=0.6, label="Generado")
    plt.title("Histograma de magnitudes")
    plt.xlabel("|h[n]|")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "histograma.png"))

    plt.figure(figsize=(8, 4))
    stride = max(1, len(metrics['freqs']) // 20)

    plt.plot(
        metrics['freqs'],
        metrics['psd_real_db'],
        label='Real',
        linewidth=2,
        marker='o',
        markersize=3,
        markevery=stride
    )

    plt.plot(
        metrics['freqs'],
        metrics['psd_fake_db'],
        label='Generado',
        linewidth=2,
        linestyle='--',
        marker='x',
        markersize=3,
        markevery=stride
    )

    plt.grid(True, alpha=0.3)
    plt.title("PSD (dB/Hz) Real vs Generado")
    plt.xlabel("Frequency (Hz)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "psd.png"))

    num_plot = min(8, args.num_eval)

    fig, axes = plt.subplots(4, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i in range(num_plot):
        stride = max(1, len(real_mag[i]) // 20)

        axes[i].plot(
            real_mag[i],
            label='Real',
            linewidth=1.8,
            marker='o',
            markersize=3,
            markevery=stride
        )

        axes[i].plot(
            fake_mag[i],
            label='Generado',
            linewidth=1.8,
            linestyle='--',
            marker='x',
            markersize=3,
            markevery=stride
        )

        axes[i].grid(True, alpha=0.3)
        axes[i].set_title(f"Señal {i + 1}", fontsize=9)
        axes[i].set_xlabel("Tap index")
        axes[i].set_ylabel("|h[n]|")
        axes[i].tick_params(axis='both', labelsize=7)

    # eliminar subplots vacíos si hay menos de 8
    for j in range(num_plot, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(
        f"Comparación señales reales vs {args.model_type.upper()}",
        fontsize=14
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    plt.tight_layout()

    plt.savefig(
        os.path.join(args.save_dir, "real_vs_generada.png")
    )

    plot_delay_distribution(args, metrics)
    plot_boxplot_delay(args, metrics)
    plot_cdf(args, real_mag, fake_mag)
    plot_kde(args, real_mag, fake_mag)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_type", type=str, choices=['gan', 'dcgan', 'wgan', 'ddpm', 'ddim'], default='dcgan')
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    config = load_config()

    args.data = config["data"]["path"]
    args.num_eval = config["evaluation"]["num_samples"]
    args.delta_tau = config["evaluation"]["delta_tau"]

    base_save_dir = config["paths"]["eval"]["base_dir"]

    # carpeta del experimento
    experiment_name = os.path.basename(
        os.path.dirname(args.model)
    )

    args.save_dir = os.path.join(
        base_save_dir,
        args.model_type.upper(),
        experiment_name
    )

    os.makedirs(args.save_dir, exist_ok=True)

    main(args, config)
