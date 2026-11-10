import math
import os
try:
    from train_models.quality import ChannelQuality
except ImportError:  # ejecutado desde dentro de train_models/
    from quality import ChannelQuality
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import json
import torch.nn.functional as F

def load_config(path="gans_config.json"):
    with open(path, "r") as f:
        return json.load(f)

# ----- GENERATOR ----- #
class Generator(nn.Module):
    def __init__(self, z_dim, L):
        super().__init__()

        self.L = int(L)
        self.init_len = int(math.ceil(self.L / 16)) # 4 unsamplings

        self.fc = nn.Linear(z_dim, 256 * self.init_len)
        self.bn0 = nn.BatchNorm1d(256)

        self.net = nn.Sequential(
            nn.ReLU(True),

            nn.ConvTranspose1d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(True),

            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),

            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(True),

            nn.Conv1d(32, 2, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.size(0), 256, self.init_len)
        x = self.bn0(x)
        x = self.net(x)
        # energy = torch.sqrt(torch.sum(x ** 2, dim=2, keepdim=True))
        # x = x / (energy + 1e-12)
        x = x[:, :, :self.L]
        return x

# ----- DISCRIMINATOR Conv1D ----- #
class Discriminator(nn.Module):
    def __init__(self, L):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv1d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv1d(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv1d(256, 512, 4, 2, 1),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Linear(512 * (L // 16), 1)

    def forward(self, x):
        f = self.net(x)
        f = f.view(x.size(0), -1)
        return self.fc(f)

### Inicialización ###
def weights_init(m):
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def gradient_penalty(critic, real, fake, device):
    # Interpolación entre datos reales y falsos
    bsize = real.size(0)
    epsilon = torch.rand(bsize, 1, 1, device=device)
    epsilon = epsilon.expand_as(real)

    interpolated = epsilon * real + (1 - epsilon) * fake
    interpolated.requires_grad_(True)

    # Salida del critic para datos interpolados
    d_interpolated = critic(interpolated)

    # Cálculo del gradiente respecto a la entrada
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    # Penalización: fuerza al gradiente a tener norma ≈ 1
    gradients = gradients.view(bsize, -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp

def compute_pdp(x):
    return torch.mean(x ** 2, dim=0)

# pérdida espectral psd
def spectral_loss(real, fake):
    real_psd = torch.log(torch.abs(torch.fft.rfft(real, dim=-1)) ** 2 + 1e-8)
    fake_psd = torch.log(torch.abs(torch.fft.rfft(fake, dim=-1)) ** 2 + 1e-8)
    return F.mse_loss(fake_psd, real_psd)

# métrica de evaluación por época
def psd_similarity(real, fake):
    """MSE entre PSDs medianas en log-escala. Cuanto menor, mejor."""
    with torch.no_grad():
        real_psd = torch.log(torch.abs(torch.fft.rfft(real, dim=-1)) ** 2 + 1e-8).mean(dim=0)
        fake_psd = torch.log(torch.abs(torch.fft.rfft(fake, dim=-1)) ** 2 + 1e-8).mean(dim=0)
        return F.mse_loss(fake_psd, real_psd).item()

def train_gan(G, C, loader, device, config, start_epoch=1, best_score=float("inf")):
    # El critic no clasifica, asigna una "energía" o score real-valued
    # Objetivo: aproximar la distancia entre distribuciones (Wasserstein)
    training = config["training"]
    paths = config["paths"]
    model_cfg = config["model"]

    os.makedirs(paths["save_dir"], exist_ok=True)

    lr = training["lr"]
    epochs = training["epochs"]
    n_critic = training["wgan"]["n_critic"]
    λ_gp = training["wgan"]["lambda_gp"]
    λ_spec = training["wgan"].get("lambda_spec", 0.1)
    patience = training["patience"]
    min_delta = training["min_delta"]

    betas = training["wgan"]["betas"]

    optC = torch.optim.Adam(C.parameters(), lr=lr, betas=betas)
    optG = torch.optim.Adam(G.parameters(), lr=lr, betas=betas)

    # El mejor modelo se elige comparando canales generados con los reales (ver quality.py)
    quality = ChannelQuality(loader.dataset.tensors[0])
    epochs_no_improve = 0

    ### Bucle del ENTRENAMIENTO principal ###
    for epoch in range(start_epoch, epochs + 1):
        c_loss_epoch = 0.0
        g_loss_epoch = 0.0
        w_dist_epoch = 0.0
        psd_score_epoch = 0.0

        for (real_batch,) in loader:
            real = real_batch.to(device)
            bsize = real.size(0)

            # WGAN usa múltiples pasos del critic por cada paso del generador
            # (el critic debe ser más fuerte para estimar bien la distancia)

            ### TRAIN CRITIC ###
            for _ in range(n_critic):
                z = torch.randn(bsize, model_cfg["z_dim"], device=device)
                fake = G(z).detach()

                real_score = C(real).mean()
                fake_score = C(fake).mean()

                # Wasserstein loss: diferencia de expectativas
                loss_C = -(real_score - fake_score)

                # Gradient penalty para imponer restricción Lipschitz
                gp = gradient_penalty(C, real, fake, device)

                loss_total = loss_C + λ_gp * gp

                optC.zero_grad()
                loss_total.backward()
                optC.step()

            # después de entrenar el critic, usamos los mismos batches de real/fake para W_dist
            with torch.no_grad():
                z = torch.randn(bsize, model_cfg["z_dim"], device=device)
                fake_for_wdist = G(z)
                # No es una loss de clasificación: es una estimación de la distancia entre distribuciones
                w_dist_epoch += (C(real).mean() - C(fake_for_wdist).mean()).item()

            ### TRAIN GENERADOR ###
            z = torch.randn(bsize, model_cfg["z_dim"], device=device)
            fake = G(z)

            loss_adv = -C(fake).mean()
            loss_spec = spectral_loss(real, fake)
            loss_G = loss_adv + λ_spec * loss_spec

            optG.zero_grad()
            loss_G.backward()
            optG.step()

            # métricas de monitoreo
            with torch.no_grad():
                z2 = torch.randn(bsize, model_cfg["z_dim"], device=device)
                fake_eval = G(z2)
                w_dist_epoch += (C(real).mean() - C(fake_eval).mean()).item()
                psd_score_epoch += psd_similarity(real, fake_eval)

            c_loss_epoch += loss_C.item()
            g_loss_epoch += loss_G.item()

        # Promedios
        c_loss_epoch /= len(loader)
        g_loss_epoch /= len(loader)
        w_dist_epoch /= len(loader)
        psd_score_epoch /= len(loader)

        q = quality.evaluate_generator(G, model_cfg["z_dim"], device)
        print(f"Epoch {epoch}/{epochs} | "
              f"W_dist={w_dist_epoch:.4f} | "
              f"PSD_score={psd_score_epoch:.4f} | "
              f"C_loss={c_loss_epoch:.4f} | "
              f"G_loss={g_loss_epoch:.4f} | "
              f"score={q['score']:.3f} (PDP {q['pdp_mae_db']:.2f} dB, KS {q['rms_ks']:.3f})", flush=True)

        if q["score"] < best_score - min_delta:
            best_score = q["score"]
            epochs_no_improve = 0

            torch.save({
                'G': G.state_dict(),
                'C': C.state_dict(),
                'epoch': epoch,
                'psd_score': psd_score_epoch,
                'score': best_score,
                'W_dist': w_dist_epoch
            }, os.path.join(paths["save_dir"], paths["best_model"]))

            print(f" Nuevo mejor modelo (score={best_score:.3f})")

        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping en epoch {epoch}")
            break

    print("Entrenamiento completado. Modelos guardados en", paths["save_dir"])

def train(config_path):
    config = load_config(config_path)
    # cada modelo guarda en su propia subcarpeta para no sobrescribir a los demás
    config["paths"]["save_dir"] = os.path.join(config["paths"]["save_dir"], "WGAN_Conv1D")
    torch.manual_seed(config["experiment"]["seed"])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = np.load(config["dataset"]["path"]).astype(np.float32)
    dataset = TensorDataset(torch.from_numpy(data))
    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        drop_last=True
    )

    G = Generator(
        config["model"]["z_dim"],
        config["model"]["signal_length"]
    ).to(device)

    C = Discriminator(
        config["model"]["signal_length"]
    ).to(device)

    best_model_path = os.path.join(
        config["paths"]["save_dir"],
        config["paths"]["best_model"]
    )

    start_epoch = 1
    best_score = float("inf")

    if os.path.exists(best_model_path):

        print(f"Cargando modelo existente: {best_model_path}")

        checkpoint = torch.load(
            best_model_path,
            map_location=device
        )

        G.load_state_dict(checkpoint["G"])
        C.load_state_dict(checkpoint["C"])

        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint.get("score", float("inf"))

        print(
            f"Reanudando desde epoch {checkpoint['epoch']} "
            f"(score={best_score:.4f})"
        )

    else:
        G.apply(weights_init)
        C.apply(weights_init)

    train_gan(G, C, loader, device, config,
              start_epoch=start_epoch, best_score=best_score)