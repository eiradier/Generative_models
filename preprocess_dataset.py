import numpy as np
import os
from matplotlib import pyplot as plt

def _load_complex_matrix(path, mat_key="est_RX_t"):
    """
    Carga una matriz compleja de canales desde .npy (complejo) o .mat (HDF5, v7.3).
    Devuelve el array tal cual viene en el archivo (sin reorientar).
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        H = np.load(path)
        if not np.iscomplexobj(H):
            raise ValueError(f"Se esperaba un array complejo en {path}, pero el dtype es {H.dtype}")
        return np.nan_to_num(H, nan=0.0)

    if ext == ".mat":
        import h5py  # solo hace falta para los .mat
        with h5py.File(path, "r") as f:
            H_struct = f[mat_key][...]
        H_real = np.nan_to_num(H_struct["real"], nan=0.0)
        H_imag = np.nan_to_num(H_struct["imag"], nan=0.0)
        return H_real + 1j * H_imag

    raise ValueError(f"Formato no soportado: {ext} (usa .npy o .mat)")


def load_loopback_data(data_path, L=200, save_path="data/datos_loopback/dataset_loopback.npy",
                       taps_axis=0, mat_key="est_RX_t", verify_dir=None):
    """
    Preprocesa medidas reales de canal loopback.

    data_path : .npy complejo (p. ej. ch_sodupe_180_time.npy) o .mat HDF5 (campo est_RX_t).
    L         : número de taps de retardo que se conservan.
    taps_axis : eje del array que corresponde a los taps de retardo.
                - .npy exportados desde MATLAB (ch_*_time.npy): (taps, canales) -> taps_axis=0
                - .mat leídos con h5py (ya traspuestos):         (canales, taps) -> taps_axis=1

    verify_dir: si se indica, guarda ahí la figura de verificación del preprocesado
                (ver verify_preprocessing).

    Devuelve y guarda un array float32 de forma (N_canales, 2, L) [real, imag] en [-1, 1].
    """
    print(f"Cargando archivo: {data_path} ...")

    H = _load_complex_matrix(data_path, mat_key=mat_key)

    print("Shape original:", H.shape)
    print("dtype:", H.dtype)

    # Orientar a (canales, taps)
    if taps_axis == 0:
        H = H.T
    elif taps_axis != 1:
        raise ValueError("taps_axis debe ser 0 o 1")

    print("Shape (canales x taps):", H.shape)

    # Comprobación: la energía debe concentrarse en los primeros taps
    pdp = np.mean(np.abs(H) ** 2, axis=0)
    frac_L = pdp[:L].sum() / (pdp.sum() + 1e-12)
    print(f"Tap con más energía media: {pdp.argmax()} | energía en los primeros {L} taps: {100 * frac_L:.2f}%")
    if frac_L < 0.5:
        print("AVISO: menos del 50% de la energía está en los primeros taps. "
              "¿Es correcto taps_axis?")

    H_full = H  # canales completos (sin recortar), para la verificación
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

    if verify_dir is not None:
        verify_preprocessing(H_full, dataset, L, save_dir=verify_dir)

    return dataset


def verify_preprocessing(H_full, dataset, L, save_dir="data/datos_loopback", example_idx=None, num_heatmap=200):
    """
    Comprueba que el preprocesado es correcto comparando las medidas originales
    (orientadas a (canales, taps), sin recortar) con el dataset final (N, 2, L).

    Figura "verificacion_preprocesado.png" (2x2):
      (a) PDP medio en dB de TODOS los taps, con el corte en L. Si los ejes estuvieran
          intercambiados no se vería un decaimiento con el retardo.
      (b) |h| en dB de canales aleatorios antes del recorte: la energía debe estar a la
          izquierda de la línea de corte.
      (c) Un canal antes y después del preprocesado (parte real e imaginaria, el original
          reescalado): las curvas deben coincidir punto a punto.
      (d) Energía de cada canal antes (original, varía) y después (debe ser constante).

    También imprime comprobaciones numéricas y lanza AssertionError si alguna falla.
    """
    os.makedirs(save_dir, exist_ok=True)
    N, L_total = H_full.shape
    assert dataset.shape == (N, 2, L), f"Shape inesperada {dataset.shape}, se esperaba {(N, 2, L)}"

    H_crop = H_full[:, :L]
    H_proc = dataset[:, 0, :].astype(np.float64) + 1j * dataset[:, 1, :].astype(np.float64)

    # Comprobaciones numéricas
    # 1) Cada canal procesado debe ser el original multiplicado por un escalar real positivo
    #    (1 / (energía * max_abs)): mismo shape temporal y misma fase.
    num = np.sum(H_proc * np.conj(H_crop), axis=1)
    escala = np.abs(num) / (np.sum(np.abs(H_crop) ** 2, axis=1) + 1e-30)
    err_rel = np.linalg.norm(H_proc - escala[:, None] * H_crop, axis=1) / (np.linalg.norm(H_proc, axis=1) + 1e-30)
    fase_media = np.angle(num)
    # 2) Energía por canal tras la normalización: debe ser la misma para todos
    energia_proc = np.sum(np.abs(H_proc) ** 2, axis=1)
    disp_energia = energia_proc.std() / energia_proc.mean()
    # 3) Fracción de energía conservada con el recorte
    pdp = np.mean(np.abs(H_full) ** 2, axis=0)
    frac_L = pdp[:L].sum() / pdp.sum()

    print("\nVERIFICACIÓN DEL PREPROCESADO")
    print(f"Canales: {N} | taps originales: {L_total} | taps conservados: {L}")
    print(f"NaN/Inf en el dataset: {int((~np.isfinite(dataset)).sum())}")
    print(f"Rango: [{dataset.min():.4f}, {dataset.max():.4f}]")
    print(f"Energía conservada en los primeros {L} taps (PDP medio): {100 * frac_L:.3f}%")
    print(f"Error relativo máx. procesado vs original reescalado: {err_rel.max():.2e}")
    print(f"Desfase máx. procesado vs original: {np.abs(fase_media).max():.2e} rad")
    print(f"Dispersión de la energía por canal tras normalizar (std/media): {disp_energia:.2e}")

    assert np.all(np.isfinite(dataset)), "El dataset contiene NaN o Inf"
    assert np.abs(dataset).max() <= 1.0 + 1e-6, "El dataset se sale de [-1, 1]"
    assert err_rel.max() < 1e-3, "Algún canal procesado no es una versión escalada del original"
    assert np.abs(fase_media).max() < 1e-3, "Algún canal procesado tiene la fase cambiada"
    assert disp_energia < 1e-3, "La energía por canal no es constante tras normalizar"
    if frac_L < 0.9:
        print(f"AVISO: solo el {100 * frac_L:.1f}% de la energía está en los primeros {L} taps")

    # Figura
    rng = np.random.default_rng(0)
    if example_idx is None:
        example_idx = int(rng.integers(N))
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # (a) PDP medio de todos los taps
    ax = axes[0, 0]
    pdp_db = 10 * np.log10(pdp / pdp.max() + 1e-20)
    ax.plot(np.arange(L_total), pdp_db)
    ax.axvline(L - 0.5, color="r", ls="--", label=f"Corte L={L}")
    ax.set_title(f"(a) PDP medio (todos los taps) - {100 * frac_L:.2f}% de energía en [0, L)")
    ax.set_xlabel("Tap (índice de retardo)")
    ax.set_ylabel("Potencia media normalizada [dB]")
    ax.legend()
    ax.grid(alpha=0.3)

    # (b) Heatmap de canales originales sin recortar
    ax = axes[0, 1]
    idx = np.sort(rng.choice(N, min(num_heatmap, N), replace=False))
    mag_db = 20 * np.log10(np.abs(H_full[idx]) / np.abs(H_full[idx]).max(axis=1, keepdims=True) + 1e-12)
    im = ax.imshow(mag_db, aspect="auto", cmap="viridis", vmin=-60, vmax=0, interpolation="nearest")
    ax.axvline(L - 0.5, color="r", ls="--")
    fig.colorbar(im, ax=ax, label="|h| normalizada [dB]")
    ax.set_title(f"(b) |h| original de {len(idx)} canales (antes del recorte)")
    ax.set_xlabel("Tap (índice de retardo)")
    ax.set_ylabel("Canal (muestra aleatoria)")

    # (c) Un canal antes / después
    ax = axes[1, 0]
    k = example_idx
    orig = escala[k] * H_crop[k]
    taps = np.arange(L)
    ax.plot(taps, orig.real, color="C0", lw=2, alpha=0.5, label="Original reescalado (Re)")
    ax.plot(taps, dataset[k, 0], "o", color="C0", ms=2.5, label="Procesado (Re)")
    ax.plot(taps, orig.imag, color="C1", lw=2, alpha=0.5, label="Original reescalado (Im)")
    ax.plot(taps, dataset[k, 1], "o", color="C1", ms=2.5, label="Procesado (Im)")
    ax.set_title(f"(c) Canal {k}: original vs procesado (error rel. {err_rel[k]:.1e})")
    ax.set_xlabel("Tap (índice de retardo)")
    ax.set_ylabel("Amplitud")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (d) Energía por canal antes y después
    ax = axes[1, 1]
    energia_orig = np.sum(np.abs(H_crop) ** 2, axis=1)
    ax.plot(10 * np.log10(energia_orig / energia_orig.mean()), lw=0.6, label="Original (primeros L taps)")
    ax.plot(10 * np.log10(energia_proc / energia_proc.mean()), lw=1.5, label="Procesado")
    ax.set_title("(d) Energía por canal relativa a la media")
    ax.set_xlabel("Índice de canal")
    ax.set_ylabel("Energía relativa [dB]")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(save_dir, "verificacion_preprocesado.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura de verificación guardada en: {out}")


def analyze_dataset(data, save_dir="data/datos_loopback", num_examples=5, heatmap_channels=100, filtrar_zeros=False, eps=1e-6):
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
    import argparse

    parser = argparse.ArgumentParser(description="Preprocesado de medidas de canal loopback")
    parser.add_argument("--input", default="data/medidas/ch_sodupe_180_time.npy",
                        help="Medidas de canal (.npy complejo o .mat HDF5)")
    parser.add_argument("--output", default="data/datos_loopback/ch_sodupe_180_time_L200.npy",
                        help="Ruta del dataset preprocesado (N, 2, L)")
    parser.add_argument("--L", type=int, default=200, help="Número de taps que se conservan")
    parser.add_argument("--taps-axis", type=int, default=0, choices=[0, 1],
                        help="Eje de los taps en el archivo de entrada (0 para los ch_*_time.npy)")
    parser.add_argument("--no-plots", action="store_true", help="No generar las figuras de análisis")
    args = parser.parse_args()

    data_loopback = load_loopback_data(
        args.input,
        L=args.L,
        save_path=args.output,
        taps_axis=args.taps_axis,
        verify_dir=None if args.no_plots else os.path.dirname(args.output),
    )

    if not args.no_plots:
        analyze_dataset(data_loopback, save_dir=os.path.dirname(args.output), filtrar_zeros=False)
