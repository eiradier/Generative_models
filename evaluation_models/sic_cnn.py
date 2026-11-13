"""
Evaluación por utilidad: estimación del canal loopback con una CNN (SRCNN modificada, Bilbao et al., "CNN-Aided
Self-Interference Estimation for In-Band Full-Duplex Systems", IEEE TCCN 11(6), 2025) entrenada con canales
de distintas fuentes (reales, clásicos, generativos) y probada siempre con canales REALES de tramos no vistos
(test de cada partición de make_segment_folds.py).

Reimplementación en Python (no se dispone del código original) del caso 4 del artículo: canal loopback
estático (medido, Sodupe) y canal directo con Doppler de 5 Hz.

Cadena de simulación (ec. 2-3 del artículo), por ventana de p símbolos OFDM consecutivos:
  - DVB-T2 8 MHz, FFT N = 2048, GI 1/8 (256 muestras), QPSK en X_F y X_L (todas las subportadoras conocidas;
    sin pilotos ni señalización L1), ISR = 10 dB (P_L / P_F), INR = 30 dB (P_L / N0).
  - H_L: CIR de 200 taps (periodo 109.375 ns = 1/fs), energía unidad, constante en la ventana (p símbolos
    ≈ p · 252 µs; la coherencia del canal medido dentro de un tramo es ~0.27 s para ρ = 0.9).
  - H_F: canal directo de la Tabla III (retardos 0, 25, 50, 75 µs; potencias −10, −30, −10, −30 dB,
    normalizadas a energía unidad), cada trayecto con desvanecimiento Rayleigh de Doppler 5 Hz (suma de 16
    sinusoides, espectro de Jakes), independiente en cada ventana.
  - Ĥ_L = H_L + sqrt(P_F/P_L) X_F H_F / X_L + N0 / X_L (frecuencia); IFFT y truncado a los 256 taps del GI.
  - Entrada de la CNN: los p símbolos truncados, [Re, Im] como 2 canales (256 × p). Salida: estimación de los
    256 taps del símbolo central.
  - Cancelación (ec. 21): C = 10 log10(ψ / θ), ψ = MSE (frecuencia) de la estimación sin filtrar, θ = MSE de
    la estimación (frecuencia, por Parseval = energía del error en los taps).

Desviaciones respecto al artículo (documentadas en PAPER_NOTES): [Re, Im] como canales en vez de concatenados
en anchura; salida = símbolo central de la entrada + corrección de la CNN en esa columna (aprendizaje
residual; como en el artículo, se estima el canal del símbolo central y los demás dan contexto); arquitectura fija = la del caso 4 en la Tabla VI del artículo (17×17/46, 3×3/23,
17×17/2 —2 salidas por [Re, Im]—, ReLU), sin búsqueda bayesiana;
número de épocas fijo, sin early stopping (para no usar datos reales de validación con fuentes sintéticas).

Estimadores comparados (el mismo ruido de test para todos):
  ls          símbolo central truncado al GI
  avg         media de los p símbolos truncados
  wiener      Wiener del artículo (de [23]): media fija de 10 subportadoras × p símbolos en frecuencia, sin
              truncar al GI (con la misma ventana de p símbolos que la CNN)
  lmmse_<s>   Wiener/LMMSE con media y covarianza de taps de la fuente s (óptimo si el canal fuese gaussiano)
  cnn_<s>     CNN entrenada con canales de la fuente s

Uso (desde la raíz, tras run de la validación cruzada por tramos):
    python evaluation_models/sic_cnn.py --fold 0
Salida: eval_results/sodupe_180_segcv/fold<k>/sic/sic_results.npz y sic_summary.txt
"""
import argparse
import os
import sys
import time
import zlib

import numpy as np
import torch
import torch.nn as nn

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
import evaluation as ev  # noqa: E402

N_FFT, N_GI = 2048, 256
FS = 8.0 / 7.0 * 8e6
T_SYM = N_FFT * (1 + 1 / 8) / FS
DIRECT_DELAYS_US = np.array([0.0, 25.0, 50.0, 75.0])
DIRECT_POW_DB = np.array([-10.0, -30.0, -10.0, -30.0])
SCALE = 16.0  # escala de entrada/salida de la CNN (taps del orden de 1)
GEN = ["gan", "dcgan", "wgan", "ddpm", "ddim"]


def unit_energy(h):
    h = np.asarray(h, dtype=np.complex128)
    return h / np.sqrt(np.sum(np.abs(h) ** 2, axis=1, keepdims=True))


def direct_channel(n, p, rng, doppler_hz, n_sin=16):
    """H_F (n, p, N_FFT): trayectos Rayleigh con Doppler (suma de sinusoides) y potencia total unidad."""
    pw = 10 ** (DIRECT_POW_DB / 10)
    pw = pw / pw.sum()
    d = np.round(DIRECT_DELAYS_US * 1e-6 * FS).astype(int)
    t = (rng.uniform(0, 100, (n, 1)) + np.arange(p)[None] * T_SYM)[..., None, None]  # (n, p, 1, 1)
    a = rng.uniform(0, 2 * np.pi, (n, 1, len(d), n_sin))
    ph = rng.uniform(0, 2 * np.pi, (n, 1, len(d), n_sin))
    g = np.exp(1j * (2 * np.pi * doppler_hz * np.cos(a) * t + ph)).sum(-1) / np.sqrt(n_sin)  # (n, p, paths)
    g = g * np.sqrt(pw)
    k = np.arange(N_FFT)
    steer = np.exp(-2j * np.pi * np.outer(d, k) / N_FFT)  # (paths, N)
    return g @ steer


def qpsk(shape, rng):
    return ((2 * rng.integers(0, 2, shape) - 1) + 1j * (2 * rng.integers(0, 2, shape) - 1)) / np.sqrt(2)


def fsmooth(H, nf):
    """Media móvil circular de nf subportadoras (última dimensión)."""
    return sum(np.roll(H, -s, axis=-1) for s in range(-(nf // 2), nf - nf // 2)) / nf


def simulate(h, p, rng, isr_db=10.0, inr_db=30.0, doppler_hz=5.0, wiener_nf=None):
    """h (n, L) complejo, energía unidad. Devuelve y (n, p, N_GI) estimaciones ruidosas truncadas al GI,
    label (n, N_GI) y psi (n,) MSE en frecuencia del símbolo central sin filtrar. Con wiener_nf, devuelve además
    el MSE (frecuencia) del Wiener del artículo: media de los p símbolos y de wiener_nf subportadoras."""
    n, L = h.shape
    hp = np.zeros((n, N_FFT), np.complex128)
    hp[:, :L] = h
    HL = np.fft.fft(hp, axis=1)[:, None, :]
    HF = direct_channel(n, p, rng, doppler_hz)
    XF, XL = qpsk((n, p, N_FFT), rng), qpsk((n, p, N_FFT), rng)
    n0 = 10 ** (-inr_db / 10)
    N0 = np.sqrt(n0 / 2) * (rng.standard_normal((n, p, N_FFT)) + 1j * rng.standard_normal((n, p, N_FFT)))
    noise = np.sqrt(10 ** (-isr_db / 10)) * XF * HF / XL + N0 / XL
    y = np.fft.ifft(HL + noise, axis=2)[..., :N_GI]
    psi = np.mean(np.abs(noise[:, p // 2]) ** 2, axis=1)
    out = (y.astype(np.complex64), hp[:, :N_GI].astype(np.complex64), psi)
    if wiener_nf:
        est = fsmooth((HL + noise).mean(1), wiener_nf)
        out += (np.mean(np.abs(est - HL[:, 0]) ** 2, axis=1),)
    return out


def noise_var_tap(p, isr_db=10.0, inr_db=30.0):
    """Varianza por tap (dominio del tiempo) del ruido de la media de p símbolos."""
    return (10 ** (-isr_db / 10) + 10 ** (-inr_db / 10)) / N_FFT / p


class SICNet(nn.Module):
    def __init__(self, c1=46, c2=23, k1=17, k2=3, k3=17):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(2, c1, k1, padding=k1 // 2), nn.ReLU(),
                                 nn.Conv2d(c1, c2, k2, padding=k2 // 2), nn.ReLU(),
                                 nn.Conv2d(c2, 2, k3, padding=k3 // 2))

    def forward(self, x):  # x (B, 2, N_GI, p)
        c = x.shape[-1] // 2  # se estima el símbolo central; los demás son contexto
        return x[..., c] + self.net(x)[..., c]  # (B, 2, N_GI)


def to_tensor(y):
    """(n, p, N_GI) complejo -> (n, 2, N_GI, p) float32 escalado."""
    y = np.transpose(y, (0, 2, 1))
    return torch.from_numpy(np.stack([y.real, y.imag], 1).astype(np.float32) * SCALE)


def label_tensor(lab):
    return torch.from_numpy(np.stack([lab.real, lab.imag], 1).astype(np.float32) * SCALE)


def train_cnn(h_train, p, epochs, rng, windows_per_epoch, batch=64, lr=1e-3, log=None):
    model = SICNet()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        idx = rng.choice(len(h_train), windows_per_epoch, replace=len(h_train) < windows_per_epoch)
        y, lab, _ = simulate(h_train[idx], p, rng)  # ruido nuevo en cada época
        x, t = to_tensor(y), label_tensor(lab)
        perm = torch.from_numpy(rng.permutation(len(x)))
        model.train()
        tot = 0.0
        for i in range(0, len(x), batch):
            j = perm[i:i + batch]
            loss = torch.mean((model(x[j]) - t[j]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(j)
        sched.step()
        if log and (ep % 10 == 0 or ep == epochs - 1):
            log(f"    época {ep + 1}/{epochs}  MSE {tot / len(x):.3e}")
    return model


@torch.no_grad()
def cnn_estimate(model, y, batch=512):
    model.eval()
    out = []
    for i in range(0, len(y), batch):
        o = model(to_tensor(y[i:i + batch])).numpy() / SCALE
        out.append(o[:, 0] + 1j * o[:, 1])
    return np.concatenate(out)


def lmmse(h_src, p):
    """Media y matriz W de LMMSE (taps del GI) estimadas con los canales de la fuente."""
    hp = np.zeros((len(h_src), N_GI), np.complex128)
    hp[:, :h_src.shape[1]] = h_src
    mu = hp.mean(0)
    C = np.cov(hp, rowvar=False)
    W = C @ np.linalg.inv(C + noise_var_tap(p) * np.eye(N_GI))
    return mu, W


def main():
    ap = argparse.ArgumentParser(description="Estimación del loopback con CNN entrenada con distintas fuentes")
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--dir", default="eval_results/sodupe_180_segcv")
    ap.add_argument("--p", type=int, default=8, help="Símbolos OFDM por ventana")
    ap.add_argument("--n-train", type=int, default=3000, help="Canales por fuente para entrenar")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--windows", type=int, default=6000, help="Ventanas (con ruido nuevo) por época")
    ap.add_argument("--test-windows", type=int, default=2, help="Ventanas de ruido por canal de test")
    ap.add_argument("--sources", nargs="+", default=["real", "real_all", "gauss", "tdl"] + GEN)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    fd = ev.repo_path(os.path.join(args.dir, f"fold{args.fold}"))
    out_dir = os.path.join(fd, "sic")
    os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.join(out_dir, "sic_log.txt"), "a", encoding="utf-8")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    cfg = ev.load_config(os.path.join(fd, "evaluation_config.json"))
    data = np.load(ev.repo_path(cfg["data"]["path"])).astype(np.float32)
    split = np.load(ev.repo_path(cfg["data"]["split_file"]))
    seg = split["segment"]
    tr, te = np.sort(split["train"]), np.sort(split["test"])
    rng = np.random.default_rng([args.seed, args.fold])
    h_train_all = unit_energy(ev.to_complex(data[tr]))
    h_test = unit_energy(ev.to_complex(data[te]))
    src = {"real_all": h_train_all,
           "real": h_train_all[np.sort(rng.choice(len(tr), min(args.n_train, len(tr)), replace=False))]}
    for m in ["gauss", "tdl"] + GEN:
        g = os.path.join(fd, "generated", f"h_generated_{m}.npy")
        if os.path.exists(g):
            src[m] = unit_energy(ev.to_complex(np.load(g)))[:args.n_train]
    sources = [s for s in args.sources if s in src]
    log(f"== Partición {args.fold}: test {len(h_test)} canales reales ({len(np.unique(seg[te]))} tramos); "
        f"p = {args.p}; fuentes {sources}")

    # Test: mismas realizaciones de ruido para todos los estimadores
    trng = np.random.default_rng([args.seed, args.fold, 999])
    h_rep = np.repeat(h_test, args.test_windows, axis=0)
    y, lab, psi, theta_w = [], [], [], []
    for i in range(0, len(h_rep), 500):  # por bloques (memoria)
        for lst, v in zip((y, lab, psi, theta_w), simulate(h_rep[i:i + 500], args.p, trng, wiener_nf=10)):
            lst.append(v)
    y, lab, psi, theta_w = (np.concatenate(v) for v in (y, lab, psi, theta_w))
    ch_index = np.repeat(np.arange(len(h_test)), args.test_windows)

    def canc(est):
        theta = np.sum(np.abs(est - lab) ** 2, axis=1)
        return 10 * np.log10(psi / theta)

    res = {"ls": canc(y[:, args.p // 2]), "avg": canc(y.mean(1)), "wiener": 10 * np.log10(psi / theta_w)}
    ybar = y.mean(1).astype(np.complex128)
    for s in sources:
        mu, W = lmmse(src[s], args.p)
        res[f"lmmse_{s}"] = canc(mu + (ybar - mu) @ W.T)
    log("  " + ", ".join(f"{k} {np.mean(v):.2f} dB" for k, v in res.items()))

    for s in sources:
        path = os.path.join(out_dir, f"cnn_{s}.pt")
        model = SICNet()
        if os.path.exists(path):
            model.load_state_dict(torch.load(path))
        else:
            t0 = time.time()
            log(f"  CNN con {s} ({len(src[s])} canales)")
            model = train_cnn(src[s], args.p, args.epochs, np.random.default_rng([args.seed, args.fold, zlib.crc32(s.encode()) % 997]),
                              args.windows, log=log)
            torch.save(model.state_dict(), path)
            log(f"    {(time.time() - t0) / 60:.1f} min")
        res[f"cnn_{s}"] = canc(cnn_estimate(model, y))
        log(f"  cnn_{s}: {np.mean(res[f'cnn_{s}']):.2f} dB")
        np.savez(os.path.join(out_dir, "sic_results.npz"), channel=ch_index, segment=seg[te][ch_index],
                 psi=psi, **res)

    lines = [f"Partición {args.fold}: cancelación media (dB) frente a canales reales de test "
             f"({len(h_test)} canales × {args.test_windows} ventanas; p = {args.p})"]
    lines += [f"  {k:<16} {np.mean(v):7.2f}  (mediana {np.median(v):.2f})" for k, v in res.items()]
    with open(os.path.join(out_dir, "sic_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log("\n".join(lines))


if __name__ == "__main__":
    main()
