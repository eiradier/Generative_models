import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.utils.data import DataLoader, TensorDataset
import os
try:
    from train_models.normalization import prepare_training_data
except ImportError:  # ejecutado desde dentro de train_models/
    from normalization import prepare_training_data
import json

def load_config(path="dm_config.json"):
    with open(path, "r") as f:
        return json.load(f)

# Denoising diffusion implicit model DDIM

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_c, out_c, 3, padding=1),
            nn.GroupNorm(8, out_c),
            nn.SiLU(),
            nn.Conv1d(out_c, out_c, 3, padding=1),
            nn.GroupNorm(8, out_c),
            nn.SiLU()
        )

    def forward(self, x):
        return self.net(x)

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2

        # frecuencias exponenciales
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)

        # codificación sinusoidal
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        return emb

class UNet1D(nn.Module):
    def __init__(self, time_emb_dim=64):
        super().__init__()

        # embedding del tiempo
        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, 256),
            nn.SiLU()
        )

        # encoder
        self.down1 = ConvBlock(2, 64)
        self.down2 = ConvBlock(64, 128)
        # reduce resolución
        self.pool = nn.MaxPool1d(2)

        # capa intermedia (representación comprimida)
        self.mid = ConvBlock(128, 256)

        # decoder
        self.up1 = ConvBlock(256 + 128, 128)
        self.up2 = ConvBlock(128 + 64, 64)

        # salida: predicción del ruido
        self.final = nn.Conv1d(64, 2, 1)

    def forward(self, x, t):
        # embedding del timestep
        t_emb = self.time_mlp(t).unsqueeze(-1)

        # encoder
        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))

        # representación comprimida
        x_mid = self.mid(self.pool(x2))

        # añadir información del tiempo
        t_emb = t_emb.expand(-1, -1, x_mid.size(2))
        x_mid = x_mid + t_emb

        # decoder
        x = F.interpolate(x_mid, size=x2.shape[-1])
        x = self.up1(torch.cat([x, x2], dim=1))

        x = F.interpolate(x, size=x1.shape[-1])
        x = self.up2(torch.cat([x, x1], dim=1))

        return self.final(x)

# SCHEDULE DE RUIDO
def cosine_beta_schedule(T):
    s = 0.008
    steps = T + 1
    x = torch.linspace(0, T, steps)

    alphas = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas = alphas / alphas[0]

    betas = 1 - (alphas[1:] / alphas[:-1])
    return torch.clip(betas, 0.0001, 0.999)

class Diffusion(nn.Module):
    def __init__(self, model, T=1000, data=None,
                 lambda_psd = 0.05):
        super().__init__()
        self.model = model

        # número total de pasos de difusión
        self.T = T
        self.lambda_psd = lambda_psd

        betas = cosine_beta_schedule(T)
        alphas = 1. - betas
        # producto acumulado
        alpha_bar = torch.cumprod(alphas, dim=0)

        # guardar constantes
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bar', alpha_bar)

        self.register_buffer('sqrt_alpha_bar', torch.sqrt(alpha_bar))
        self.register_buffer('sqrt_one_minus', torch.sqrt(1 - alpha_bar))

        if data is None:
            raise ValueError("data must be provided to compute tap_weights")

        mag = np.abs(data[:, 0, :] + 1j * data[:, 1, :])
        pdp = np.mean(mag ** 2, axis=0)  # (L,)
        pdp_norm = pdp / (pdp.max() + 1e-12)

        tap_weights = 1.0 + 2.0 * pdp_norm
        tap_weights = torch.tensor(tap_weights, dtype=torch.float32)

        self.register_buffer('tap_weights', tap_weights)

    # FORWARD DIFFUSION
    # genera xt a partir de x0
    # mezcla señal original + ruido gaussiano
    def q_sample(self, x0, t, noise):
        return (
            self.sqrt_alpha_bar[t][:, None, None] * x0 +
            self.sqrt_one_minus[t][:, None, None] * noise
        )

    # FUNCIÓN DE PÉRDIDA
    # la red aprende a predecir el ruido añadido
    # objetivo: epsilon_theta(xt, t) = epsilon
    def loss(self, x0):
        b = x0.size(0)

        # elegir paso aleatorio
        t = torch.randint(0, self.T, (b,), device=x0.device)
        # ruido gaussiano
        noise = torch.randn_like(x0)

        # generar señal ruidosa
        xt = self.q_sample(x0, t, noise)

        # predicción del ruido
        noise_pred = self.model(xt, t)

        # pesos con forma (1, 1, L) para broadcasting
        weights = self.tap_weights.view(1, 1, -1)
        loss_noise = (weights * (noise_pred - noise) ** 2).mean()

        a_t = self.alpha_bar[t][:, None, None]
        x0_pred = (xt - torch.sqrt(1 - a_t) * noise_pred) / torch.sqrt(a_t + 1e-8)
        x0_pred = torch.clamp(x0_pred, -1, 1)

        # pérdida espectral, penaliza desviaciones en PSD en log-escala
        real_psd = torch.abs(torch.fft.rfft(x0, dim=-1)) ** 2
        pred_psd = torch.abs(torch.fft.rfft(x0_pred, dim=-1)) ** 2
        real_psd = real_psd / (real_psd.sum(dim=-1, keepdim=True) + 1e-8)
        pred_psd = pred_psd / (pred_psd.sum(dim=-1, keepdim=True) + 1e-8)
        loss_psd = F.mse_loss(torch.log1p(pred_psd), torch.log1p(real_psd))

        loss = loss_noise + self.lambda_psd * loss_psd

        return loss

# Generar nuevas señales a partir de ruido puro
class DDIMSampler:
    def __init__(self, diffusion, eta=0.0):
        self.model = diffusion.model
        self.T = diffusion.T

        # controla aleatoriedad, eta=0 proceso determinista
        self.eta = eta
        self.alpha_bar = diffusion.alpha_bar

    @torch.no_grad()
    def sample(self, shape, device, steps=50):
        # comenzar desde ruido gaussiano puro
        x = torch.randn(shape, device=device)
        # seleccionar subconjunto de timesteps
        time_steps = torch.linspace(self.T - 1, 0, steps, device=device).long()

        for i in range(len(time_steps) - 1):
            t = time_steps[i]
            t_next = time_steps[i + 1]
            t_batch = torch.full((shape[0],), t, device=device)

            # estimar ruido presente en la señal
            eps = self.model(x, t_batch)

            a_t = self.alpha_bar[t].view(1, 1, 1)
            a_next = self.alpha_bar[t_next].view(1, 1, 1)

            # estimación de la señal limpia x0 = (xt - ruido) / escala
            x0_pred = (x - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t + 1e-8)
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

            # controla cantidad de ruido extra
            sigma = self.eta * torch.sqrt(
                (1 - a_next) / (1 - a_t + 1e-8)
                * (1 - a_t / (a_next + 1e-8))
            )

            noise = torch.randn_like(x) if self.eta > 0 else 0

            # ecuación DDIM: combina señal estimada limpia, dirección del ruido y ruido opcional
            x = (
                torch.sqrt(a_next) * x0_pred +
                torch.sqrt(torch.clamp(1 - a_next - sigma ** 2, min=0)) * eps +
                sigma * noise
            )

        # último paso: señal limpia
        # en t=0: extrae x0 limpio
        t_last = time_steps[-1]
        t_batch = torch.full((shape[0],), t_last, device=device)
        eps = self.model(x, t_batch)
        a_t = self.alpha_bar[t_last].view(1, 1, 1)
        x0_final = (x - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t + 1e-8)
        x0_final = torch.clamp(x0_final, -1.0, 1.0)

        return x0_final

def train(config_path):
    config = load_config(config_path)
    # cada modelo guarda en su propia subcarpeta para no sobrescribir a los demás
    config["paths"]["save_dir"] = os.path.join(config["paths"]["save_dir"], "DDIM")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(config["paths"]["save_dir"], exist_ok=True)

    data_np, _, _ = prepare_training_data(config, config["paths"]["save_dir"])

    data = torch.from_numpy(data_np).float()

    loader = DataLoader(
        TensorDataset(data),
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"]
    )

    model = UNet1D(time_emb_dim=config["model"]["time_emb_dim"]).to(device)

    diffusion = Diffusion(
        model,
        T=config["diffusion"]["T"],
        data=data_np,
        lambda_psd = config["diffusion"].get("lambda_psd", 0.05),
    ).to(device)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["lr"]
    )

    best_loss = float('inf')
    epochs_no_improve = 0

    # Reanudación: last.pth guarda el estado completo al final de cada época
    last_path = os.path.join(config["paths"]["save_dir"], "last.pth")
    start_epoch = 1
    if os.path.exists(last_path):
        st = torch.load(last_path, map_location=device, weights_only=False)
        if st.get("finished"):
            print(f"El entrenamiento ya había terminado (época {st['epoch']}); no se repite.")
            return
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start_epoch = st["epoch"] + 1
        best_loss = st["best_loss"]
        epochs_no_improve = st["epochs_no_improve"]
        print(f"Reanudando desde la época {st['epoch']} (mejor loss {best_loss:.6f})")

    def save_last(epoch, finished=False):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch,
                    "best_loss": best_loss, "epochs_no_improve": epochs_no_improve,
                    "finished": finished}, last_path)

    epoch = start_epoch - 1

    for epoch in range(start_epoch, config["training"]["epochs"] + 1):
        loss_avg = 0.0

        for (x,) in loader:
            x = x.to(device)

            loss = diffusion.loss(x)

            opt.zero_grad()
            loss.backward()

            # evita el nan por exploding gradient
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config["training"]["grad_clip"]
            )

            opt.step()
            loss_avg += loss.item()

        loss_avg /= len(loader)

        print(f"Epoch {epoch} | Loss {loss_avg:.6f}")

        if loss_avg < best_loss - config["training"]["min_delta"]:
            best_loss = loss_avg
            epochs_no_improve = 0

            save_path = os.path.join(
                config["paths"]["save_dir"],
                config["paths"]["best_model"]
            )

            torch.save(model.state_dict(), save_path)
            print(f"Nuevo mejor modelo guardado | epoch {epoch} | loss {loss_avg:.6f} -> {save_path}")
        else:
            epochs_no_improve += 1

        save_last(epoch)

        if epochs_no_improve >= config["training"]["patience"]:
            print(f"\nEarly stopping")
            break

    save_last(epoch, finished=True)
