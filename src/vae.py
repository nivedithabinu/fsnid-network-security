import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BetaVAE(nn.Module):
    """
    Beta-VAE for Network Intrusion Detection Feature Selection.
    
    Architecture:
    Encoder:
      x[41] -> Linear(41, 128) -> ReLU
            -> Linear(128, 64) -> ReLU
            -> Linear(64, 10)  -> z_mean[10]
            -> Linear(64, 10)  -> z_log_var[10]
            
    Reparameterize:
      z = z_mean + eps * exp(0.5 * z_log_var), eps ~ N(0, I)
      
    Decoder:
      z[10] -> Linear(10, 64)  -> ReLU
            -> Linear(64, 128) -> ReLU
            -> Linear(128, 41) -> x_reconstructed[41]
            
    ELBO Loss:
      recon_loss = MSE(x_reconstructed, x)
      kl_loss = -0.5 * sum(1 + z_log_var - z_mean^2 - exp(z_log_var))
      total_loss = recon_loss + beta * kl_loss (beta = 4)
    """
    def __init__(self, input_dim=41, latent_dim=10, beta=4.0):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta

        # Encoder
        self.encoder_shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True)
        )
        self.fc_mean = nn.Linear(64, latent_dim)
        self.fc_log_var = nn.Linear(64, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, input_dim)
        )

    def encode(self, x):
        h = self.encoder_shared(x)
        z_mean = self.fc_mean(h)
        z_log_var = self.fc_log_var(h)
        return z_mean, z_log_var

    def reparameterize(self, z_mean, z_log_var):
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return z_mean + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        x_recon = self.decode(z)
        return x_recon, z_mean, z_log_var

    def elbo_loss(self, x_recon, x, z_mean, z_log_var):
        """
        Computes negative ELBO with beta weighting.
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_recon, x, reduction="mean")
        
        # KL Divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        # Normalized per sample and per feature to balance nicely with MSE
        kl_element = 1.0 + z_log_var - z_mean.pow(2) - z_log_var.exp()
        kl_loss = -0.5 * torch.mean(torch.sum(kl_element, dim=1)) / self.input_dim

        total_loss = recon_loss + self.beta * kl_loss
        return total_loss, recon_loss, kl_loss

    def compute_elbo(self, X, batch_size=1024):
        """
        Compute average ELBO loss (scalar) over dataset X.
        """
        self.eval()
        if not isinstance(X, torch.Tensor):
            X_tensor = torch.tensor(X, dtype=torch.float32)
        else:
            X_tensor = X.float()

        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        total_loss = 0.0
        n_samples = 0

        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(device)
                x_recon, z_mean, z_log_var = self.forward(batch_x)
                loss, _, _ = self.elbo_loss(x_recon, batch_x, z_mean, z_log_var)
                total_loss += loss.item() * batch_x.size(0)
                n_samples += batch_x.size(0)

        return total_loss / n_samples

    def fit(self, X_train, epochs=200, batch_size=512, lr=0.001, verbose=True):
        """
        Train Beta-VAE on X_train.
        """
        self.to(device)
        self.train()

        if not isinstance(X_train, torch.Tensor):
            X_tensor = torch.tensor(X_train, dtype=torch.float32)
        else:
            X_tensor = X_train.float()

        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            total_loss = 0.0
            for (batch_x,) in loader:
                batch_x = batch_x.to(device)
                optimizer.zero_grad()
                x_recon, z_mean, z_log_var = self.forward(batch_x)
                loss, _, _ = self.elbo_loss(x_recon, batch_x, z_mean, z_log_var)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch_x.size(0)

            if verbose and (epoch + 1) % 40 == 0:
                print(f"[Beta-VAE] Epoch {epoch + 1:3d}/{epochs} | Loss: {total_loss / len(X_tensor):.6f}")

        return self
