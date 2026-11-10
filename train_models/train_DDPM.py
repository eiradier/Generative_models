import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.utils.data import DataLoader, TensorDataset
import os

def load_config(path="dm_config.json"):
    with open(path, "r") as f:
        return json.load(f)

# Denoising diffusion probabilistic model DDPM
class ConvBlock(nn.Module):
    """
    Bloque básico Conv1D para la U-Net
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_c, out_c, 3, padding=1),
            nn.GroupNorm(min(8, out_c), num_channels=out_c),  # normalización estable para batch pequeños
            nn.SiLU(),
            nn.Conv1d(out_c, out_c, 3, padding=1),
            nn.GroupNorm(min(8, out_c), num_channels=out_c),
            nn.SiLU()
        )

    def forward(self, x):
        return self.net(x)

class TimeEmbedding(nn.Module):
    """Embedding sinusoidal del timestep"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        # Frecuencias logarítmicamente espaciadas
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        # embedding sinusoidal tipo Transformer
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        # salida: (batch, dim)
        return emb

class UNet1D(nn.Module):
    def __init__(self, time_emb_dim=64):
        super().__init__()
        # MLP que transforma el timestep t en un embedding
        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, 256),
            nn.SiLU()
        )
        # Encoder (downsampling)
        # extrae características a distintas escalas temporales
        self.down1 = ConvBlock(2, 64)
        self.down2 = ConvBlock(64, 128)
        self.pool = nn.MaxPool1d(2)
        # Bottleneck
        # parte central donde se inyecta la información temporal
        self.mid = ConvBlock(128, 256)
        # Decoder (unsampling)
        # reconstruye la señal combinando información fina y gruesa
        self.up1 = ConvBlock(256 + 128, 128)
        self.up2 = ConvBlock(128 + 64, 64)
        # Capa final: predicción del ruido (misma dimensión que la señal)
        self.final = nn.Conv1d(64, 2, 1)

    def forward(self, x, t):
        # Encoder
        x1 = self.down1(x) # características finas
        x2 = self.down2(self.pool(x1)) # características más globales

        # Bottleneck + time conditioning
        x_mid = self.mid(self.pool(x2))
        # expandir embeddings temporal a la dimensión temporal de x_mid
        t_emb = self.time_mlp(t).unsqueeze(-1)
        t_emb = t_emb.expand(-1, -1, x_mid.size(2))
        x_mid = x_mid + t_emb

        # Decoder: reconstrucción progresiva
        x = F.interpolate(x_mid, size=x2.shape[-1])
        x = self.up1(torch.cat([x, x2], dim=1))

        x = F.interpolate(x, size=x1.shape[-1])
        x = self.up2(torch.cat([x, x1], dim=1))

        # predicción final del ruido ε
        return self.final(x)

def cosine_beta_schedule(T):
    """
    Define cuánto ruido se añade en cada paso de difusión.
    """
    s = 0.008
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas = alphas / alphas[0]
    # beta_t = 1 - alpha_t / alpha_{t-1}
    betas = 1 - (alphas[1:] / alphas[:-1])
    return torch.clip(betas, 0.0001, 0.999)

class DDPM(nn.Module):
    """
    - proceso forward (añadir ruido)
    - función de pérdida
    """
    def __init__(self, model, T=1000):
        super().__init__()
        self.model = model
        self.T = T
        # Definición del proceso de difusión
        betas = cosine_beta_schedule(T)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        # Guardar constantes como buffers (no entrenables)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alpha', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus', torch.sqrt(1 - alphas_cumprod))

    def q_sample(self, x0, t, noise):
        """
        Aplicar el proceso forward:
        """
        # Añade ruido a la señal original según el timestep t
        # x_t = mezcla de señal limpia + ruido
        return (
            self.sqrt_alpha[t][:, None, None] * x0 +
            self.sqrt_one_minus[t][:, None, None] * noise
        )

    def loss(self, x0):
        """
        Función de pérdida
        """
        b = x0.size(0)
        # timestep aleatorio por muestra
        t = torch.randint(0, self.T, (b,), device=x0.device)
        # ruido gaussiano ε ~ N(0, I)
        noise = torch.randn_like(x0)
        # señal ruidosa x_t
        xt = self.q_sample(x0, t, noise)
        # predicción del ruido
        noise_pred = self.model(xt, t)
        # MSE loss
        return F.mse_loss(noise_pred, noise)

def train(config_path):
    """
    Bucle principal de entrenamiento del DDPM.
    """
    config = load_config(config_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(config["paths"]["save_dir"], exist_ok=True)
    # cargar dataset
    data_np = np.load(config["dataset"]["path"])
    data = torch.from_numpy(data_np).float()

    loader = DataLoader(
        TensorDataset(data),
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"]
    )

    model = UNet1D(time_emb_dim=config["model"]["time_emb_dim"]).to(device)
    ddpm = DDPM(model, T=config["diffusion"]["T"]).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])

    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, config["training"]["epochs"] + 1):
        loss_avg = 0.0

        for (x,) in loader:
            # x = señal real
            x = x.to(device)
            loss = ddpm.loss(x) # aprendizaje de denoising

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip"])
            opt.step()

            loss_avg += loss.item()

        loss_avg /= len(loader)
        print(f"Epoch {epoch} | Loss {loss_avg:.6f}")

        if loss_avg < best_loss - config["training"]["min_delta"]:
            best_loss = loss_avg
            epochs_no_improve = 0

            torch.save(model.state_dict(),
                os.path.join(config["paths"]["save_dir"], config["paths"]["best_model"]))

            print(f"Mejor modelo guardado (loss={best_loss:.6f})")
        else:
            epochs_no_improve += 1
            print(f"Sin mejora durante {epochs_no_improve} epochs")

        if epochs_no_improve >= config["training"]["patience"]:
            print(f"Early stopping")
            break