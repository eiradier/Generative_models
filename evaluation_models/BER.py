import numpy as np

def qpsk_mapper(bits):
    # Agrupar bits de 2 en 2 (QPSK usa 2 bits por símbolo)
    bits = bits.reshape(-1, 2)
    # Convertir bits a valores ±1
    # 0 → +1, 1 → -1
    symbols = (1 - 2*bits[:, 0]) + 1j * (1 - 2*bits[:, 1])
    # Normalizar energía del símbolo
    return symbols / np.sqrt(2)

def qpsk_demapper(symbols):
    # Si parte real es negativa → bit 1
    bits_real = (np.real(symbols) < 0).astype(int)
    # Si parte imaginaria es negativa → bit 1
    bits_imag = (np.imag(symbols) < 0).astype(int)
    # Recombinar en vector de bits plano
    return np.stack([bits_real, bits_imag], axis=1).reshape(-1)

def add_awgn(signal, snr_db):
    # Convertir SNR de dB a escala lineal
    snr = 10 ** (snr_db / 10)
    # Potencia media de la señal
    power = np.mean(np.abs(signal) ** 2)
    # Potencia del ruido según SNR
    noise_power = power / snr

    # Ruido complejo gaussiano (real + imaginario)
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(*signal.shape) +
        1j * np.random.randn(*signal.shape)
    )
    # Señal + ruido
    return signal + noise

def compute_ber(h, bits, mod_type="QPSK"):
    N, L = h.shape
    num_symbols = N * L

    if mod_type == "QPSK":
        bits = bits[:num_symbols * 2].reshape(-1, 2)
        x = np.zeros(len(bits), dtype=complex)
        x[(bits[:,0]==0)&(bits[:,1]==0)] = 1 + 1j
        x[(bits[:,0]==0)&(bits[:,1]==1)] = 1 - 1j
        x[(bits[:,0]==1)&(bits[:,1]==0)] = -1 + 1j
        x[(bits[:,0]==1)&(bits[:,1]==1)] = -1 - 1j
        x /= np.sqrt(2)

    else:  # 16QAM
        bits = bits[:num_symbols * 4].reshape(-1, 4)
        mapping = {(0,0):-3, (0,1):-1, (1,1):1, (1,0):3}
        x = np.array([
            mapping[tuple(b[:2])] + 1j * mapping[tuple(b[2:])]
            for b in bits
        ])
        x /= np.sqrt(10)

    x = x.reshape(N, L)

    H = np.fft.fft(h, axis=0)
    y = H * x

    # AWGN
    snr_db = 10
    snr = 10**(snr_db/10)
    noise_power = np.mean(np.abs(y)**2) / snr
    noise = np.sqrt(noise_power/2) * (
        np.random.randn(*y.shape) + 1j*np.random.randn(*y.shape)
    )

    y += noise

    # ZF equalization
    x_hat = y / (H + 1e-12)

    x_hat = x_hat.reshape(-1)

    # demapper
    if mod_type == "QPSK":
        rx_bits = np.stack([
            (np.real(x_hat) < 0),
            (np.imag(x_hat) < 0)
        ], axis=1).reshape(-1)

        tx_bits = bits.reshape(-1)

    else:
        x_hat = x_hat * np.sqrt(10)
        real = np.real(x_hat)
        imag = np.imag(x_hat)

        rx_bits = np.stack([
            (real > 0),
            (np.abs(real) < 2),
            (imag > 0),
            (np.abs(imag) < 2)
        ], axis=1).reshape(-1)

        tx_bits = bits.reshape(-1)

    return np.mean(rx_bits != tx_bits)