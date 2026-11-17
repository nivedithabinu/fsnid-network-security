import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Generator(nn.Module):
    """
    Generator network per minority class.
    Architecture:
    noise[100] -> Linear(100, 128) -> BatchNorm1d(128) -> LeakyReLU(0.2)
               -> Linear(128, 256) -> BatchNorm1d(256) -> LeakyReLU(0.2)
               -> Linear(256, 41)  -> Tanh -> synthetic_sample[41]
    """
    def __init__(self, noise_dim=100, output_dim=41):
        super().__init__()
        self.noise_dim = noise_dim
        self.output_dim = output_dim

        self.net = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    """
    Shared Discriminator network.
    Architecture:
    sample[41] -> Linear(41, 128) -> LeakyReLU(0.2) -> Dropout(0.3)
               -> Linear(128, 64) -> LeakyReLU(0.2) -> Dropout(0.3)
               -> Linear(64, 1)   -> Sigmoid -> real/fake
    """
    def __init__(self, input_dim=41):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class IntegratedClassifier(nn.Module):
    """
    Integrated auxiliary classifier.
    Architecture:
    sample[41] -> Linear(41, 64) -> ReLU -> Linear(64, 5) -> LogSoftmax -> class[5]
    """
    def __init__(self, input_dim=41, num_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)


class TMG_GAN:
    """
    Tabular Multi-Generator Generative Adversarial Network (TMG-GAN).
    Reference: Ding et al., IEEE T-IFS 2024.
    
    Generates synthetic samples specifically for minority classes (R2L: class 3, U2R: class 4)
    using dedicated generators, a shared discriminator, and an integrated classifier.
    """
    def __init__(
        self,
        input_dim=41,
        num_classes=5,
        noise_dim=100,
        minority_classes=(3, 4),
        lambda1=1.0,
        lambda2=0.5,
        lr=0.0002,
        batch_size=256
    ):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.noise_dim = noise_dim
        self.minority_classes = list(minority_classes)
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lr = lr
        self.batch_size = batch_size

        # Networks
        self.generators = {
            cls_id: Generator(noise_dim=self.noise_dim, output_dim=self.input_dim).to(device)
            for cls_id in self.minority_classes
        }
        self.discriminator = Discriminator(input_dim=self.input_dim).to(device)
        self.classifier = IntegratedClassifier(input_dim=self.input_dim, num_classes=self.num_classes).to(device)

        # Optimizers
        self.g_optimizers = {
            cls_id: optim.Adam(self.generators[cls_id].parameters(), lr=self.lr, betas=(0.5, 0.999))
            for cls_id in self.minority_classes
        }
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.c_optimizer = optim.Adam(self.classifier.parameters(), lr=self.lr, betas=(0.5, 0.999))

        # Losses
        self.adv_criterion = nn.BCELoss()
        self.cls_criterion = nn.NLLLoss()

        # Class centroids in [-1, 1] range for cosine similarity
        self.class_centroids = {}

    def _to_gan_scale(self, x):
        # Maps [0, 1] to [-1, 1] to align with Tanh output
        return 2.0 * x - 1.0

    def _to_data_scale(self, x):
        # Maps [-1, 1] back to [0, 1]
        return np.clip((x + 1.0) / 2.0, 0.0, 1.0)

    def train(self, X_train, y_train, epochs=200, verbose=True):
        """
        Train TMG-GAN on training data.
        X_train: (N, 41) in [0, 1] range
        y_train: (N,) integer class labels [0..4]
        """
        X_scaled = self._to_gan_scale(X_train)

        # Compute and store class centroids for minority classes
        for cls_id in self.minority_classes:
            mask = (y_train == cls_id)
            if np.sum(mask) > 0:
                mean_vec = np.mean(X_scaled[mask], axis=0)
                self.class_centroids[cls_id] = torch.tensor(mean_vec, dtype=torch.float32).to(device)
            else:
                self.class_centroids[cls_id] = torch.zeros(self.input_dim, dtype=torch.float32).to(device)

        # Prepare DataLoader
        train_X_t = torch.tensor(X_scaled, dtype=torch.float32)
        train_y_t = torch.tensor(y_train, dtype=torch.long)
        dataset = TensorDataset(train_X_t, train_y_t)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        for epoch in range(epochs):
            self.discriminator.train()
            self.classifier.train()
            for g in self.generators.values():
                g.train()

            total_d_loss = 0.0
            total_g_loss = 0.0

            for real_x, real_y in dataloader:
                real_x = real_x.to(device)
                real_y = real_y.to(device)
                b_size = real_x.size(0)

                real_labels = torch.ones(b_size, 1, device=device)
                fake_labels = torch.zeros(b_size, 1, device=device)

                # =========================================================
                # 1. TRAIN DISCRIMINATOR & INTEGRATED CLASSIFIER
                # =========================================================
                self.d_optimizer.zero_grad()
                self.c_optimizer.zero_grad()

                # Discriminator on real data
                d_real = self.discriminator(real_x)
                d_real_loss = self.adv_criterion(d_real, real_labels)

                # Classifier on real data
                c_real_pred = self.classifier(real_x)
                c_real_loss = self.cls_criterion(c_real_pred, real_y)

                # Discriminator on generated data from each minority generator
                d_fake_losses = []
                for cls_id, generator in self.generators.items():
                    z = torch.randn(b_size // len(self.minority_classes), self.noise_dim, device=device)
                    fake_x = generator(z)
                    d_fake = self.discriminator(fake_x.detach())
                    fake_target = torch.zeros(fake_x.size(0), 1, device=device)
                    d_fake_losses.append(self.adv_criterion(d_fake, fake_target))

                d_fake_loss = torch.mean(torch.stack(d_fake_losses))
                d_loss = d_real_loss + d_fake_loss + self.lambda1 * c_real_loss

                d_loss.backward()
                self.d_optimizer.step()
                self.c_optimizer.step()

                total_d_loss += d_loss.item()

                # =========================================================
                # 2. TRAIN EACH MINORITY GENERATOR
                # =========================================================
                for cls_id, generator in self.generators.items():
                    self.g_optimizers[cls_id].zero_grad()

                    z = torch.randn(b_size, self.noise_dim, device=device)
                    gen_x = generator(z)

                    # Adversarial loss: fool the discriminator
                    d_gen = self.discriminator(gen_x)
                    g_adv_loss = self.adv_criterion(d_gen, real_labels)

                    # Classification loss: force generated sample to belong to cls_id
                    c_gen_pred = self.classifier(gen_x)
                    target_cls = torch.full((b_size,), cls_id, dtype=torch.long, device=device)
                    g_cls_loss = self.cls_criterion(c_gen_pred, target_cls)

                    # Cosine similarity loss: align with real minority class prototype
                    centroid = self.class_centroids[cls_id].unsqueeze(0)  # (1, 41)
                    cos_sim = F.cosine_similarity(gen_x, centroid, dim=1)
                    cos_loss = 1.0 - torch.mean(cos_sim)

                    # Total generator loss
                    g_loss = g_adv_loss + self.lambda1 * g_cls_loss + self.lambda2 * cos_loss
                    g_loss.backward()
                    self.g_optimizers[cls_id].step()

                    total_g_loss += g_loss.item()

            if verbose and (epoch + 1) % 40 == 0:
                print(f"[TMG-GAN] Epoch {epoch + 1:3d}/{epochs} | D Loss: {total_d_loss/len(dataloader):.4f} | G Loss: {total_g_loss/len(dataloader):.4f}")

    def generate(self, class_id, n_samples):
        """
        Generate synthetic samples for a specific minority class.
        Returns: np.ndarray of shape (n_samples, 41) in [0, 1] range.
        """
        if class_id not in self.generators:
            raise ValueError(f"No generator available for class {class_id}. Available: {list(self.generators.keys())}")

        generator = self.generators[class_id]
        generator.eval()

        samples = []
        batch_size = 512
        num_generated = 0

        with torch.no_grad():
            while num_generated < n_samples:
                cur_batch = min(batch_size, n_samples - num_generated)
                z = torch.randn(cur_batch, self.noise_dim, device=device)
                fake_x = generator(z).cpu().numpy()
                samples.append(fake_x)
                num_generated += cur_batch

        samples = np.concatenate(samples, axis=0)
        # Convert [-1, 1] back to [0, 1]
        return self._to_data_scale(samples)

    def balance(self, X_train, y_train, target_counts=None):
        """
        Balances the training set using TMG-GAN.
        Default targets per roadmap:
          U2R (class 4): 500 samples
          R2L (class 3): 2,000 samples
        Returns: (X_balanced, y_balanced)
        """
        if target_counts is None:
            target_counts = {4: 500, 3: 2000}

        synthetic_X_list = [X_train]
        synthetic_y_list = [y_train]

        for cls_id, target_n in target_counts.items():
            current_n = np.sum(y_train == cls_id)
            needed = target_n - current_n
            if needed > 0:
                print(f"[TMG-GAN] Generating {needed} synthetic samples for class {cls_id} (current={current_n} -> target={target_n})...")
                synth_x = self.generate(cls_id, needed)
                synth_y = np.full((needed,), cls_id, dtype=y_train.dtype)
                synthetic_X_list.append(synth_x)
                synthetic_y_list.append(synth_y)
            else:
                print(f"[TMG-GAN] Class {cls_id} already has {current_n} >= target {target_n}, no generation needed.")

        X_balanced = np.concatenate(synthetic_X_list, axis=0)
        y_balanced = np.concatenate(synthetic_y_list, axis=0)

        # Shuffle balanced dataset
        shuffle_idx = np.random.permutation(len(X_balanced))
        X_balanced = X_balanced[shuffle_idx]
        y_balanced = y_balanced[shuffle_idx]

        print(f"[TMG-GAN] Balanced dataset shape: {X_balanced.shape}, class counts: {np.bincount(y_balanced)}")
        return X_balanced, y_balanced
