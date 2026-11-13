import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# β-VAE
# =========================================================

class BetaVAE(nn.Module):
    def __init__(self, input_dim=41, latent_dim=10, beta=4):
        super().__init__()
        
        self.beta       = beta
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.fc_mean    = nn.Linear(64, latent_dim)
        self.fc_log_var = nn.Linear(64, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    # ----------------------------------------------------------
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mean(h), self.fc_log_var(h)

    def reparameterize(self, z_mean, z_log_var):
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return z_mean + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z_mean, z_log_var = self.encode(x)
        z      = self.reparameterize(z_mean, z_log_var)
        x_recon = self.decode(z)
        return x_recon, z_mean, z_log_var

    # ----------------------------------------------------------
    def elbo_loss(self, x):
        """
        Compute ELBO = reconstruction loss + β × KL divergence.
        Returns total loss, recon_loss, kl_loss.
        """
        x_recon, z_mean, z_log_var = self.forward(x)

        recon_loss = nn.functional.mse_loss(
            x_recon, x, reduction="mean"
        )
        kl_loss = -0.5 * torch.mean(
            1 + z_log_var - z_mean.pow(2) - z_log_var.exp()
        )

        total = recon_loss + self.beta * kl_loss
        return total, recon_loss, kl_loss

    # ----------------------------------------------------------
    def compute_elbo(self, X_numpy):
        """
        Compute scalar ELBO on a numpy array (no gradient).
        Used by vae_fs.py for feature importance scoring.
        """
        self.eval()
        with torch.no_grad():
            X_t    = torch.tensor(X_numpy, dtype=torch.float32).to(device)
            loss, _, _ = self.elbo_loss(X_t)
        return loss.item()


# =========================================================
# TRAINING FUNCTION
# =========================================================

def train_vae(
    X_train,
    input_dim=41,
    latent_dim=10,
    beta=4,
    epochs=200,
    batch_size=256,
    lr=1e-3
):
    """
    Train a β-VAE on X_train and return the trained model.

    Parameters
    ----------
    X_train    : np.ndarray (n, input_dim)
    input_dim  : number of features (41 for NSL-KDD)
    latent_dim : size of latent space (default 10)
    beta       : weight for KL divergence (default 4)
    epochs     : training epochs (default 200)
    batch_size : mini-batch size (default 256)
    lr         : Adam learning rate (default 1e-3)

    Returns
    -------
    model : trained BetaVAE
    """
    model     = BetaVAE(input_dim, latent_dim, beta).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_t    = torch.tensor(X_train, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(X_t),
        batch_size=batch_size,
        shuffle=True
    )

    print(
        f"\nTraining β-VAE | input_dim={input_dim} | "
        f"latent_dim={latent_dim} | β={beta} | epochs={epochs}"
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            loss, _, _ = model.elbo_loss(batch_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 50 == 0:
            print(
                f"  Epoch {epoch+1:3d}/{epochs} | "
                f"ELBO: {total_loss/len(loader):.4f}"
            )

    print("β-VAE training complete.")
    return model
