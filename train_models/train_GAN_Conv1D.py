import os
import numpy as np
import torch
from torch import nn
from torch.nn.utils import spectral_norm
from torch.utils.data import TensorDataset, DataLoader
import json

def load_config(path="gans_config.json"):
    with open(path, "r") as f:
        return json.load(f)

# ----- GENERATOR ----- #
class Generator(nn.Module):
    def __init__(self, z_dim, L):
        super().__init__()
        self.init_len = L // 8
        # Proyección del vector de ruido z a un espacio más grande
        self.fc = nn.Linear(z_dim, 128 * self.init_len)
        self.net = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.ConvTranspose1d(128, 64, 4, 2, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),

            nn.ConvTranspose1d(64, 32, 4, 2, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(True),

            nn.ConvTranspose1d(32, 2, 4, 2, 1),
            nn.Tanh()
        )
    def forward(self, z):
        # z: vector aleatorio (ruido): entrada del generador
        x = self.fc(z)
        x = x.view(z.size(0), 128, self.init_len)
        # genera señal de canal sintético
        return self.net(x)

# ----- DISCRIMINATOR ----- #
class Discriminator(nn.Module):
    def __init__(self, L):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Conv1d(2, 32, 4, 2, 1)),
            nn.LeakyReLU(0.2),

            spectral_norm(nn.Conv1d(32, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2),

            spectral_norm(nn.Conv1d(64, 128, 4, 2, 1)),
            nn.LeakyReLU(0.2),
        )
        # Capa final: clasificación binaria (real vs falso)
        self.fc = spectral_norm(nn.Linear(128 * (L // 8), 1))

    def forward(self, x):
        # x: señal real o generada
        f = self.net(x)
        f = f.view(x.size(0), -1)
        return self.fc(f)

def weights_init(m):
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def train_gan(G, D, loader, device, config):
    training = config["training"]
    model_cfg = config["model"]
    paths = config["paths"]

    os.makedirs(paths["save_dir"], exist_ok=True)

    epochs = training["epochs"]
    lr = training["lr"]
    min_delta = training["min_delta"]
    patience = training["patience"]
    z_dim = model_cfg["z_dim"]

    betas = training["gan"]["betas"]

    # Función de pérdida binaria (real vs falso)
    criterion = nn.BCEWithLogitsLoss()

    # Optimizadores independientes para G y D
    optD = torch.optim.Adam(D.parameters(), lr=lr, betas=betas)
    optG = torch.optim.Adam(G.parameters(), lr=lr, betas=betas)

    # Mejor pérdida del generador (para guardar mejor modelo)
    best_g_loss = float("inf")
    epochs_no_improve = 0

    ### Bucle del ENTRENAMIENTO principal ###
    for epoch in range(epochs):
        g_loss_epoch = 0.0
        d_loss_epoch = 0.0

        for (real_batch,) in loader:
            real = real_batch.to(device)
            bsize = real.size(0)

            ### 1) TRAIN DISCRIMINADOR ###
            optD.zero_grad()
            # Generar muestras falsas
            z = torch.randn(bsize, z_dim, device=device)
            fake = G(z).detach()
            # detach() evita actualizar el generador en esta fase

            # Predicciones del discriminador
            pred_real = D(real) # debe clasificar como 1
            pred_fake = D(fake) # debe clasificar como 0

            # Etiquetas reales y falsas
            real_labels = torch.ones((bsize, 1), device=device)
            fake_labels = torch.zeros((bsize, 1), device=device)

            # Pérdida del discriminador:
            # - aprender a distinguir real vs falso
            loss_real = criterion(pred_real, real_labels)
            loss_fake = criterion(pred_fake, fake_labels)

            lossD = 0.5 * (loss_real + loss_fake)
            lossD.backward()
            optD.step()

            ### 2) TRAIN GENERADOR ###
            optG.zero_grad()

            # Generar nuevas muestras falsas
            z = torch.randn(bsize, z_dim, device=device)
            fake = G(z)

            # El generador intenta engañar al discriminador:
            pred_fake = D(fake)
            lossG = criterion(pred_fake, real_labels)

            lossG.backward()
            optG.step()

            # Acumular pérdidas
            g_loss_epoch += lossG.item()
            d_loss_epoch += lossD.item()

        g_loss_epoch /= len(loader)
        d_loss_epoch /= len(loader)

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"G_loss: {g_loss_epoch:.4f} | D_loss: {d_loss_epoch:.4f}"
        )

        # Guardar mejor modelo (según pérdida del generador)
        if g_loss_epoch < best_g_loss - min_delta:
            best_g_loss = g_loss_epoch
            epochs_no_improve = 0

            torch.save(
                {
                    "G": G.state_dict(),
                    "D": D.state_dict(),
                    "epoch": epoch,
                    "g_loss": best_g_loss
                },
                os.path.join(paths["save_dir"], paths["best_model"]),
            )

            print(f"Nuevo mejor modelo en {epoch}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping en época {epoch} (sin mejora en {patience} épocas)")
            break

    print("Entrenamiento completado. Modelos guardados en", paths["save_dir"])

def train(config_path):
    config = load_config(config_path)
    # cada modelo guarda en su propia subcarpeta para no sobrescribir a los demás
    config["paths"]["save_dir"] = os.path.join(config["paths"]["save_dir"], "GAN_Conv1D")
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

    D = Discriminator(
        config["model"]["signal_length"]
    ).to(device)

    G.apply(weights_init)
    D.apply(weights_init)

    train_gan(G, D, loader, device, config)
