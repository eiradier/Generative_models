import numpy as np
import os
import h5py
from matplotlib import pyplot as plt

def load_loopback_data(mat_path, L=200, save_path="data/dataset_loopback.npy"):
    print(f"Cargando archivo: {mat_path} ...")

    f = h5py.File(mat_path, "r")

    H_struct = f["est_RX_t"][...]

    print("Shape original:", H_struct.shape)
    print("dtype:", H_struct.dtype)

    H_real = np.nan_to_num(H_struct["real"], nan=0.0)
    H_imag = np.nan_to_num(H_struct["imag"], nan=0.0)

    H = H_real + 1j * H_imag

    print("Shape complejo:", H.shape)

    H = H[:, :L]
    print(f"Shape tras recorte a L={L}:", H.shape)

    H_real = np.real(H)
    H_imag = np.imag(H)

    energy = np.sqrt(np.sum(H_real**2 + H_imag**2, axis=1, keepdims=True))

    H_real = H_real / (energy + 1e-12)
    H_imag = H_imag / (energy + 1e-12)

    dataset = np.stack([H_real, H_imag], axis=1).astype(np.float32)

    print("Shape dataset final:", dataset.shape)

    max_abs = np.max(np.abs(dataset))
    dataset = dataset / (max_abs + 1e-12)

    print("Rango final:", dataset.min(), dataset.max())

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    np.save(save_path, dataset)

    print("Dataset guardado en:", save_path)

    return dataset


def load_synthetic_data(file_path, L=128, save_path="data/datos_sinteticos/TDL_D_85ns_fd50000_SNR20_h_real.npy"):
    print(f"Cargando archivo: {file_path}")
    H = np.load(file_path) # (128, 100000)
    print("Shape original:", H.shape)
    print("dtype:", H.dtype)
    print("Primeros valores:\n", H[:5, :5])
    print("Min:", H.min(), "Max:", H.max())

    H= H[:L, :]
    H = H.T # (100000, 128)
    print(f"Shape canales: {H.shape}")

    # 1. normalización física del canal
    power = np.mean(np.abs(H) ** 2, axis=1, keepdims=True)
    H = H / np.sqrt(power + 1e-12)

    # 2. evita outliers
    scale = np.percentile(np.abs(H), 99)
    H = H / (scale + 1e-12)

    # 3. pasar a real/imag
    dataset = np.stack([H.real, H.imag], axis=1)
    print("Shape dataset final:", dataset.shape)

    max_abs = np.max(np.abs(dataset))
    dataset = dataset / (max_abs + 1e-12)
    print("Rango final:", dataset.min(), dataset.max())

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, dataset)
    print("Dataset guardado en:", save_path)

    return dataset

def load_nist_data(mat_path, L=128, save_path="data/datos_reales/dataset_nist.npy"):
    print(f"Cargando archivo {mat_path} ...")
    f = h5py.File(mat_path, "r")

    print("Claves encontradas:")
    keys = list(f.keys())
    for k in keys:
        print(" ", k)

    # extraer todas las matrices que empiezan por "h_"
    # channel_keys = [k for k in keys if k.startswith("h_")]

    # print("\nMatrices de canal encontradas:")
    # for ck in channel_keys:
    #     print(" ", ck)

    channel_keys = ["h_AAplant_5G"]

    if channel_keys[0] not in keys:
        raise ValueError(f"No se encontró {channel_keys[0]} en el archivo .mat")

    print("\nMatriz de canal seleccionada:")
    print(" ", channel_keys[0])

    all_channels = []
    total_channels = 0

    # Procesar cada matriz h_XXXX
    for ck in channel_keys:
        print(f"\nProcesando {ck} ...")

        H_raw = f[ck] # matriz 241 x YYY
        # conservar el dtype estructurado
        H_struct = H_raw[...]

        print("dtype:", H_struct.dtype)

        # reconstruir números complejos
        H_real = np.nan_to_num(H_struct["real"], nan=0.0)
        H_imag = np.nan_to_num(H_struct["imag"], nan=0.0)
        H = H_real + 1j * H_imag

        print("Shape original:", H.shape)

        # H es 241 x num_canales, a veces se invierte dependiendo de h5py
        if H.shape[0] != 241 and H.shape[1] == 241:
            print("La matriz está transpuesta. Corrigiendo orientación...")
            H = H.T

        # H es 241 x YYY, YYY = número de canales
        print("Shape actual (debería ser 241 x N):", H.shape)

        # recortar a 128 muestras
        H_crop = H[:L, :] # shape: L x num_canales
        print("Shape tras recorte a L=128:", H_crop.shape)
        # convertir columnas -> filas
        H_crop = H_crop.T
        print(" Shape (canales x L):", H_crop.shape)

        num_c = H_crop.shape[0]
        total_channels += num_c

        # Separar real e imaginario (SIN normalizar todavía)
        H_real = np.real(H_crop)
        H_imag = np.imag(H_crop)

        # Normalización
        energy = np.sqrt(np.sum(H_real**2 + H_imag**2, axis=1, keepdims=True))
        H_real = H_real / (energy + 1e-12)
        H_imag = H_imag / (energy + 1e-12)


        H_2_channel = np.stack([H_real, H_imag], axis=1)
        all_channels.append(H_2_channel.astype(np.float32))

        print(f"Añadidos {num_c} canales")

    # juntar todos los canales
    dataset = np.vstack(all_channels)

    print("\n" + "=" * 60)
    print("RESUMEN FINAL DEL DATASET")
    print("=" * 60)
    print("Canales totales:", total_channels)
    print("Shape final:", dataset.shape)

    # # [-1, 1] normalización
    max_abs = np.max(np.abs(dataset))
    dataset = dataset / (max_abs + 1e-12)

    print("Max absoluto global:", max_abs)
    print("Rango final:", dataset.min(), dataset.max())

    # guardar
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, dataset)

    print(f"\nDataset guardado en: {save_path}")
    return dataset

def analyze_dataset(data, save_dir="data/datos_sinteticos", num_examples=5, heatmap_channels=100, filtrar_zeros=False, eps=1e-6):
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    N, C, L = data.shape
    assert C == 2, "El dataset debe tener 2 canales (real, imag)"

    print("\nDATASET ANALYSIS")
    print(f"Total de canales: {N}")
    print(f"Canales por muestra: {C}")
    print(f"Longitud temporal: {L}")
    print(f"Min global: {data.min():.4f} | Max global: {data.max():.4f}")

    real = data[:, 0, :]
    imag = data[:, 1, :]
    magnitud = np.sqrt(real ** 2 + imag ** 2)

    real_flat = real.flatten()
    imag_flat = imag.flatten()
    mag_flat  = magnitud.flatten()

    # Máscara de taps activos (se usa solo si filtrar_zeros=True)
    if filtrar_zeros:
        mask = mag_flat > eps
        real_plot = real_flat[mask]
        imag_plot = imag_flat[mask]
        mag_plot  = mag_flat[mask]
        print(f"Taps activos: {mask.sum()} / {len(mask)} ({mask.mean() * 100:.2f}%)")
        sufijo_titulo = " (taps activos)"
    else:
        real_plot = real_flat
        imag_plot = imag_flat
        mag_plot  = mag_flat
        sufijo_titulo = ""

    # Ejemplos de canales (real e imaginario)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Parte REAL
    for i in range(min(num_examples, N)):
        axes[0].plot(real[i], alpha=0.7)

    axes[0].set_title("Ejemplos - Parte REAL")
    axes[0].set_xlabel("Muestra temporal")
    axes[0].set_ylabel("Amplitud")
    axes[0].grid(alpha=0.3)

    # Parte IMAGINARIA
    for i in range(min(num_examples, N)):
        axes[1].plot(imag[i], alpha=0.7)

    axes[1].set_title("Ejemplos - Parte IMAGINARIA")
    axes[1].set_xlabel("Muestra temporal")
    axes[1].set_ylabel("Amplitud")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if save_dir:
        plt.savefig(os.path.join(save_dir, "examples_real_imag.png"), dpi=300, bbox_inches="tight")
        plt.close()

    # Histograma Real / Imag
    p_low  = np.percentile(real_plot, 0.5)
    p_high = np.percentile(real_plot, 99.5)
    plt.figure(figsize=(6, 4))
    plt.hist(real_plot, bins=100, density=True, alpha=0.6, label="Real", range=(p_low, p_high))
    plt.hist(imag_plot, bins=100, density=True, alpha=0.6, label="Imag", range=(p_low, p_high))
    plt.title(f"Distribución global Real / Imag{sufijo_titulo}")
    plt.xlabel("Valor normalizado")
    plt.ylabel("Densidad")
    plt.legend()
    plt.grid(alpha=0.3)
    if save_dir:
        plt.savefig(os.path.join(save_dir, "histogram_real_imag.png"), dpi=300, bbox_inches="tight")
        plt.close()

    # Histograma magnitud + perfil por tap
    p1  = np.percentile(mag_plot, 1)
    p99 = np.percentile(mag_plot, 99)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(mag_plot, bins=100, density=True, alpha=0.8, range=(p1, p99))
    axes[0].set_title(f"Distribución de |H|{sufijo_titulo}")
    axes[0].set_xlabel("Magnitud")
    axes[0].set_ylabel("Densidad")
    axes[0].grid(alpha=0.3)

    mean_mag_per_tap = magnitud.mean(axis=0)
    axes[1].plot(mean_mag_per_tap)
    axes[1].set_title("Magnitud media por tap de retardo")
    axes[1].set_xlabel("Tap (índice de retardo)")
    axes[1].set_ylabel("Magnitud media")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "histogram_magnitud.png"), dpi=300, bbox_inches="tight")
        plt.close()

    # Perfil temporal medio
    mean_real = real.mean(axis=0)
    std_real = real.std(axis=0)
    mean_imag = imag.mean(axis=0)
    std_imag = imag.std(axis=0)

    # Rango Y dinámico: excluye outliers del tap dominante
    y_zoom = np.percentile(np.abs(mean_real + std_real), 95) * 2
    y_zoom = max(y_zoom, 0.01)  # mínimo razonable para canales muy planos

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, mean, std, label in [
        (axes[0], mean_real, std_real, "Media Real"),
        (axes[1], mean_imag, std_imag, "Media Imag"),
    ]:
        ax.plot(mean, label=label)
        ax.fill_between(np.arange(L), mean - std, mean + std, alpha=0.3)
        ax.set_xlabel("Muestra temporal")
        ax.legend()
        ax.grid(alpha=0.3)

    axes[0].set_title("Perfil temporal medio - Parte Real")
    axes[1].set_title("Perfil temporal medio - Parte Imaginaria")

    # Solo aplicar zoom si el canal es sparse (hay tap dominante claro)
    if filtrar_zeros:
        axes[0].set_ylim(-y_zoom, y_zoom)
        axes[1].set_ylim(-y_zoom, y_zoom)

    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "mean_profile.png"), dpi=300, bbox_inches="tight")
        plt.close()

    # Heatmap magnitud
    idx    = np.random.choice(N, min(heatmap_channels, N), replace=False)
    subset = magnitud[idx]
    plt.figure(figsize=(8, 6))
    plt.imshow(subset, aspect="auto", cmap="viridis")
    plt.colorbar(label="Magnitud")
    plt.title(f"Heatmap de magnitud ({subset.shape[0]} canales)")
    plt.xlabel("Muestra temporal")
    plt.ylabel("Canal")
    if save_dir:
        plt.savefig(os.path.join(save_dir, "heatmap_magnitud.png"), dpi=300, bbox_inches="tight")
        plt.close()

    print("\nAnálisis completado.")

if __name__ == "__main__":
    data_sintetico = load_synthetic_data(
        "data/datos_sinteticos/TDL_D_85ns_fd1000_SNR20_h_real.npy",
        L=128,
        save_path="data/datos_sinteticos/dataset_synthetic.npy"
    )
    data_loopback = load_loopback_data(
        "data/datos_loopback/new_ch_time_worst.mat",
        L=200,
        save_path="data/datos_loopback/new_ch_time_worst.npy"
    )
    data_nist = load_nist_data(
        "data/datos_NIST/NIST_Samples.mat",
        L=128,
        save_path="data/datos_NIST/h_AAplant_5G.npy"
    )

    analyze_dataset(data_sintetico, save_dir="data/datos_sinteticos", filtrar_zeros=True,  eps=1e-6)
    analyze_dataset(data_loopback,  save_dir="data/datos_loopback",   filtrar_zeros=False)
    analyze_dataset(data_nist, save_dir="data/datos_NIST/h_AAplant_5G", filtrar_zeros=True,  eps=1e-4)
