import os
try:
    from train_models.normalization import prepare_training_data
except ImportError:  # ejecutado desde dentro de train_models/
    from normalization import prepare_training_data
try:
    from train_models.quality import ChannelQuality
except ImportError:  # ejecutado desde dentro de train_models/
    from quality import ChannelQuality
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
        self.fc = nn.Linear(z_dim, 128 * self.init_len)
        self.net = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.ConvTranspose1d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),

            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(True),

            nn.ConvTranspose1d(32, 2, kernel_size=3, padding=1),
            nn.Tanh()
        )
    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.size(0), 128, self.init_len)
        x = self.net(x)
        return x

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
        # Capa final que convierte las características en un único valor
        self.fc = spectral_norm(nn.Linear(128 * (L // 8), 1))

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

def train_gan(G, D, loader, device, config, data_orig, scaler):
    model_cfg = config["model"]
    paths = config["paths"]
    training = config["training"]
    epochs = training["epochs"]
    z_dim = model_cfg["z_dim"]
    lr = training["lr"]
    min_delta = training["min_delta"]
    patience = training["patience"]
    save_dir = paths["save_dir"]

    os.makedirs(save_dir, exist_ok=True)

    # Loss tipo regresión (en vez de clasificación)
    # El discriminador aprende a dar valores cercanos a:
    # 1 señal real
    # 0 señal falsa
    criterion = nn.MSELoss()

    # optimizadores Adam para G y D
    optD = torch.optim.Adam(D.parameters(), lr=lr, betas=training["gan"]["betas"])
    optG = torch.optim.Adam(G.parameters(), lr=lr, betas=training["gan"]["betas"])

    # El mejor modelo se elige comparando canales generados con los reales (ver quality.py)
    quality = ChannelQuality(data_orig)
    best_score = float('inf')
    epochs_no_improve = 0

    ### Bucle del ENTRENAMIENTO principal ###
    for epoch in range(1, epochs + 1):
        g_loss_avg = 0.0
        d_loss_avg = 0.0

        for (real_batch,) in loader:
            real = real_batch.to(device)
            bsize = real.size(0)

            ### 1) TRAIN DISCRIMINADOR ###
            optD.zero_grad()

            # Se añade un pequeño ruido a las señales reales
            # para mejorar la robustez del discriminador
            real_noisy = real + 0.001 * torch.randn_like(real)

            # Etiquetas suavizadas (mejor estabilidad):
            # reales → 0.9 en vez de 1
            # falsas → 0.1 en vez de 0 CAMBIADO
            real_labels = torch.full((bsize, 1), 0.9, device=device)
            fake_labels = torch.full((bsize, 1), 0.0, device=device)

            # Señales reales
            pred_real = D(real_noisy)
            loss_real = criterion(pred_real, real_labels)

            # Señales falsas (no actualiza G)
            z = torch.randn(bsize, z_dim, device=device)
            fake = G(z).detach()

            pred_fake = D(fake)
            loss_fake = criterion(pred_fake, fake_labels)

            # El discriminador aprende a diferenciar real vs falso
            lossD = 0.5 * (loss_real + loss_fake)
            lossD.backward()
            optD.step()

            ### TRAIN GENERADOR ###
            optG.zero_grad()

            # Genera nuevas señales desde ruido
            z = torch.randn(bsize, z_dim, device=device)
            fake = G(z)

            # El generador quiere que el discriminador piense que
            # estas señales son reales (0.9 ~ 1)
            pred_fake = D(fake)
            real_labels = torch.full((bsize, 1), 0.9, device=device)

            lossG = criterion(pred_fake, real_labels)
            lossG.backward()
            optG.step()

            # acumular pérdidas
            g_loss_avg += lossG.item()
            d_loss_avg += lossD.item()

        # promedios por época
        g_loss_avg /= len(loader)
        d_loss_avg /= len(loader)

        q = quality.evaluate_generator(G, z_dim, device, transform=scaler.inverse)
        print(f"Epoch {epoch}/{epochs} | G_loss={g_loss_avg:.4f} | D_loss={d_loss_avg:.4f} | "
              f"score={q['score']:.3f} (PDP {q['pdp_mae_db']:.2f} dB, KS {q['rms_ks']:.3f})")

        # Guardar mejor modelo (según la calidad de los canales generados)
        if q["score"] < best_score - min_delta:
            best_score = q["score"]
            epochs_no_improve = 0
            torch.save(
                {
                    'G': G.state_dict(),
                    'D': D.state_dict(),
                    'epoch': epoch,
                    'G_loss': g_loss_avg,
                    'D_loss': d_loss_avg,
                    'score': best_score
                },
                os.path.join(save_dir, 'model_best.pth')
            )
            print(f"Nuevo mejor modelo en la época {epoch} | score={best_score:.3f}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping en época {epoch} (sin mejora en {patience} épocas)")
            break

    print("Entrenamiento completado. Modelos guardados en", save_dir)
    print(f"Mejor score: {best_score:.3f}")

def train(config_path):
    config = load_config(config_path)
    # cada modelo guarda en su propia subcarpeta para no sobrescribir a los demás
    config["paths"]["save_dir"] = os.path.join(config["paths"]["save_dir"], "DCGAN_Conv1D")
    torch.manual_seed(config["experiment"]["seed"])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(config["paths"]["save_dir"], exist_ok=True)
    data, data_orig, scaler = prepare_training_data(config, config["paths"]["save_dir"])
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

    train_gan(G, D, loader, device, config, data_orig, scaler)
