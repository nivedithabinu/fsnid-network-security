import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# GENERATOR
# One per minority class — learns that class's distribution
# =========================================================

class Generator(nn.Module):

    def __init__(self, latent_dim=100, output_dim=41):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),

            # No activation — discriminator constrains output range
            nn.Linear(256, output_dim)
        )

    def forward(self, z):
        return self.net(z)


# =========================================================
# DISCRIMINATOR
# Shared — distinguishes real from synthetic samples
# =========================================================

class Discriminator(nn.Module):

    def __init__(self, input_dim=41):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# =========================================================
# INTEGRATED CLASSIFIER
# Helps generators produce class-specific samples
# =========================================================

class IntegratedClassifier(nn.Module):

    def __init__(self, input_dim=41, num_classes=5):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# =========================================================
# TMG-GAN
# =========================================================

class TMG_GAN:

    def __init__(self, input_dim=41, num_classes=5, latent_dim=100):
        self.input_dim   = input_dim
        self.num_classes = num_classes
        self.latent_dim  = latent_dim

        self.minority_classes = []
        self.generators = {}

        self.discriminator = Discriminator(input_dim).to(device)
        self.classifier    = IntegratedClassifier(input_dim, num_classes).to(device)

        # Store data range for inverse-scaling
        self.data_min = None
        self.data_max = None

    # ----------------------------------------------------------
    def _scale(self, X):
        """Scale data to [-1, 1] for GAN training."""
        self.data_min = X.min(axis=0)
        self.data_max = X.max(axis=0)
        denom = self.data_max - self.data_min
        denom[denom == 0] = 1.0
        return 2.0 * (X - self.data_min) / denom - 1.0

    def _unscale(self, X_scaled):
        """Inverse-scale from [-1, 1] back to original range."""
        denom = self.data_max - self.data_min
        denom[denom == 0] = 1.0
        return (X_scaled + 1.0) / 2.0 * denom + self.data_min

    # ----------------------------------------------------------
    def train(
        self,
        X_train,
        y_train,
        minority_classes,
        epochs=200,
        batch_size=64,
        lr=0.0002,
        lambda1=1.0,
        lambda2=0.5
    ):
        """
        Train TMG-GAN.

        Parameters
        ----------
        X_train          : np.ndarray (n, 41)
        y_train          : np.ndarray (n,) integer labels
        minority_classes : list of class indices to oversample, e.g. [3, 4]
        epochs           : training epochs
        batch_size       : mini-batch size
        lr               : Adam learning rate
        lambda1          : weight for classification loss
        lambda2          : weight for cosine similarity loss
        """
        self.minority_classes = minority_classes

        X_scaled = self._scale(X_train.copy().astype(np.float32))

        # One generator per minority class
        for cls in minority_classes:
            self.generators[cls] = Generator(
                self.latent_dim, self.input_dim
            ).to(device)

        # Optimizers
        d_opt = optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999)
        )
        c_opt = optim.Adam(
            self.classifier.parameters(), lr=lr, betas=(0.5, 0.999)
        )
        g_opts = {
            cls: optim.Adam(
                self.generators[cls].parameters(), lr=lr, betas=(0.5, 0.999)
            )
            for cls in minority_classes
        }

        adv_fn = nn.BCELoss()
        cls_fn = nn.CrossEntropyLoss()

        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        y_t = torch.tensor(y_train,  dtype=torch.long)

        loader = DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )

        print(
            f"\nTraining TMG-GAN | epochs={epochs} | "
            f"minority classes={minority_classes}"
        )

        for epoch in range(epochs):

            d_ep, g_ep = 0.0, 0.0

            for real_x, real_y in loader:

                real_x = real_x.to(device)
                real_y = real_y.to(device)
                n = real_x.size(0)

                real_lbl = torch.ones(n,  1).to(device)
                fake_lbl = torch.zeros(n, 1).to(device)

                # ── Discriminator + Classifier ──────────────────
                d_opt.zero_grad()
                c_opt.zero_grad()

                d_loss = adv_fn(self.discriminator(real_x), real_lbl)
                c_loss = cls_fn(self.classifier(real_x), real_y)

                d_fake_loss = 0.0
                for cls in minority_classes:
                    z = torch.randn(n, self.latent_dim).to(device)
                    fake_x = self.generators[cls](z).detach()
                    d_fake_loss += adv_fn(self.discriminator(fake_x), fake_lbl)
                d_fake_loss /= len(minority_classes)

                d_total = d_loss + d_fake_loss + lambda1 * c_loss
                d_total.backward()
                d_opt.step()
                c_opt.step()

                d_ep += d_total.item()

                # ── Generators ──────────────────────────────────
                for cls in minority_classes:

                    g_opts[cls].zero_grad()

                    z     = torch.randn(n, self.latent_dim).to(device)
                    fake_x = self.generators[cls](z)

                    # Adversarial loss
                    g_adv = adv_fn(self.discriminator(fake_x), real_lbl)

                    # Classification loss
                    cls_lbl = torch.full(
                        (n,), cls, dtype=torch.long
                    ).to(device)
                    g_cls = cls_fn(self.classifier(fake_x), cls_lbl)

                    # Cosine similarity loss — push generators apart
                    g_cos = torch.tensor(0.0, device=device)
                    if len(minority_classes) > 1:
                        sims, cnt = 0.0, 0
                        for other in minority_classes:
                            if other != cls:
                                z2 = torch.randn(n, self.latent_dim).to(device)
                                with torch.no_grad():
                                    fx2 = self.generators[other](z2)
                                sims += F.cosine_similarity(
                                    fake_x, fx2, dim=1
                                ).mean()
                                cnt += 1
                        g_cos = sims / cnt

                    g_total = g_adv + lambda1 * g_cls + lambda2 * g_cos
                    g_total.backward()
                    g_opts[cls].step()

                    g_ep += g_total.item()

            if (epoch + 1) % 50 == 0:
                print(
                    f"  Epoch {epoch+1:3d}/{epochs} | "
                    f"D: {d_ep/len(loader):.4f} | "
                    f"G: {g_ep/len(loader):.4f}"
                )

        print("TMG-GAN training complete.")

    # ----------------------------------------------------------
    def generate(self, class_id, n_samples):
        """
        Generate n_samples for class_id.
        Returns np.ndarray in original data range.
        """
        if class_id not in self.generators:
            raise ValueError(
                f"No generator for class {class_id}. "
                f"Trained classes: {list(self.generators.keys())}"
            )

        self.generators[class_id].eval()
        with torch.no_grad():
            z       = torch.randn(n_samples, self.latent_dim).to(device)
            samples = self.generators[class_id](z).cpu().numpy()

        return self._unscale(samples)

    # ----------------------------------------------------------
    def balance(self, X_train, y_train, target_counts=None):
        """
        Oversample minority classes to target_counts.

        Parameters
        ----------
        target_counts : dict {class_id: n_target}
                        default → {3: 2000, 4: 500}

        Returns
        -------
        X_balanced, y_balanced : np.ndarray (shuffled)
        """
        if target_counts is None:
            target_counts = {3: 2000, 4: 500}

        X_bal = list(X_train)
        y_bal = list(y_train)

        print("\nBalancing dataset with TMG-GAN...")
        for cls in self.minority_classes:
            current = int(np.sum(y_train == cls))
            target  = target_counts.get(cls, 500)

            if current < target:
                n_gen = target - current
                synth = self.generate(cls, n_gen)
                X_bal.extend(synth)
                y_bal.extend([cls] * n_gen)
                print(
                    f"  Class {cls}: {current} → {target} "
                    f"(+{n_gen} synthetic samples)"
                )
            else:
                print(
                    f"  Class {cls}: {current} samples, "
                    f"target={target} already met."
                )

        X_balanced = np.array(X_bal, dtype=np.float32)
        y_balanced = np.array(y_bal, dtype=np.int64)

        perm = np.random.permutation(len(X_balanced))
        return X_balanced[perm], y_balanced[perm]
