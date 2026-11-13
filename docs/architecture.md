# System Architecture — TMG-GAN + VAE-FSNID

## Overview

The project implements a two-phase pipeline:
- **Phase 1**: Replication of FSNID (KDD 2025) on NSL-KDD
- **Phase 2**: Extension with TMG-GAN data balancing and β-VAE feature selection

---

## Phase 1 Architecture (FSNID Replication)

```
NSL-KDD Dataset (125,973 samples, 41 features, 5 classes)
                        │
                   80:20 split (seed=42)
           ┌───────────┴────────────┐
      X_train (100,778)         X_test (25,195)
           │
    ┌──────┴───────────────┐
    │  FSNID Feature       │
    │  Selection (MINE)    │
    │                      │
    │  For each feature i: │
    │  MI(X_with_i, Y)     │
    │  - MI(X_without_i,Y) │
    │  = Φ_i               │
    │                      │
    │  Null model threshold│
    │  Keep if Φ_i > null  │
    └──────┬───────────────┘
           │ 12 features selected
           │ [4,11,12,15,18,20,23,26,27,31,36,38]
           ▼
    MLP Classifier
    input → 250 → 250 → 5 (LogSoftmax)
    SGD lr=0.1, momentum=0.9, epochs=100
           │
    Results: 86.78% accuracy, 84.43% F1
```

---

## Phase 2 Architecture (Extension)

```
NSL-KDD Training Set (100,778 samples, 41 features)
Normal:53858  DoS:36720  Probe:9362  R2L:798  U2R:40
                        │
            ┌───────────┴──────────────────┐
            │   STAGE 1: TMG-GAN           │
            │   Data Balancing             │
            │                              │
            │   Generator_R2L (cls=3)      │
            │   noise[100] → 256 → 256→41  │
            │                              │
            │   Generator_U2R (cls=4)      │
            │   noise[100] → 256 → 256→41  │
            │                              │
            │   Shared Discriminator       │
            │   41 → 256 → 128 → 1        │
            │                              │
            │   Integrated Classifier      │
            │   41 → 128 → 5              │
            │                              │
            │   Loss = Adversarial         │
            │        + λ1×Classification   │
            │        + λ2×CosineSimilarity │
            │                              │
            │   R2L:  798 → 2,000         │
            │   U2R:   40 →   500         │
            └───────────┬──────────────────┘
                        │ Balanced X_train (~102,440 samples)
                        │
            ┌───────────┴──────────────────┐
            │   STAGE 2: β-VAE             │
            │   Feature Selection          │
            │                              │
            │   Encoder:                   │
            │   41→128→64→(μ[10],σ[10])   │
            │                              │
            │   Reparameterize:            │
            │   z = μ + ε×exp(0.5σ)       │
            │                              │
            │   Decoder:                   │
            │   10→64→128→41              │
            │                              │
            │   ELBO = MSE + β×KL  (β=4) │
            │                              │
            │   For each feature i:        │
            │   shuffle column i           │
            │   ΔELBO_i = ELBO_masked      │
            │           - ELBO_baseline    │
            │                              │
            │   Null threshold:            │
            │   10 random shuffles         │
            │   threshold = mean + 2×std   │
            │                              │
            │   Keep if ΔELBO_i > thresh  │
            └───────────┬──────────────────┘
                        │ VAE-selected features
                        │
            ┌───────────┴──────────────────┐
            │   STAGE 3: MLP Classifier    │
            │                              │
            │   input → 250 → 250 → 5     │
            │   SGD lr=0.01, momentum=0.9  │
            │   50 epochs, batch=1024      │
            │   3 seeds + 95% CI          │
            └───────────┬──────────────────┘
                        │
                 Final Results Table
```

---

## File-Level Architecture

### `src/tmg_gan.py`
- **`Generator`** (`nn.Module`): `latent[100] → Linear(256,BN,LReLU) × 2 → output[41]`
- **`Discriminator`** (`nn.Module`): `input[41] → Linear(256,LReLU,Drop) × 2 → sigmoid[1]`
- **`IntegratedClassifier`** (`nn.Module`): `input[41] → Linear(128,ReLU) → Linear[5]`
- **`TMG_GAN`** (class): orchestrates training with 3-part generator loss
  - `train(X, y, minority_classes, epochs, batch_size)`
  - `generate(class_id, n_samples)` → `np.ndarray`
  - `balance(X, y, target_counts)` → balanced `(X, y)`

### `src/vae.py`
- **`BetaVAE`** (`nn.Module`):
  - `encode(x)` → `(z_mean, z_log_var)`
  - `reparameterize(z_mean, z_log_var)` → `z`
  - `decode(z)` → `x_recon`
  - `elbo_loss(x)` → `(total, recon, kl)`
  - `compute_elbo(X_numpy)` → scalar (used by `vae_fs.py`)
- **`train_vae(X_train, ...)`** → trained `BetaVAE`

### `src/vae_fs.py`
- **`vae_feature_selection(X_train, ...)`**:
  1. Trains `BetaVAE` on `X_train`
  2. Computes baseline ELBO
  3. Estimates null threshold via 10 random column shuffles
  4. Scores each feature by ΔELBO when that column is permuted
  5. Returns `(selected_indices, vae_model, importances)`

### `src/classifier.py`
- **`FeatureClassifier`** (`nn.Module`): `input → Linear(250,ReLU) → Linear(250,ReLU) → Linear(5,LogSoftmax)`

### `src/fsnid_fs.py`
- **`fsnid_selection`** (class): wraps `mine_fa` for MINE-based feature selection
  - `null_model()` → random noise MI threshold
  - `mi_ordering()` → sort features by individual MI
  - `run_main()` → selected feature indices

### `src/mine.py`
- **`mine`**: KL-divergence based Mutual Information estimator
- **`mine_fa`**: faster classification-based MI estimator (used by FSNID)

---

## Ablation Study Design

| Config | Balancing | Feature Selection | Key Question |
|---|---|---|---|
| A | None | All 41 | Full-feature upper bound |
| B | None | FSNID-MINE (12) | Does MINE selection hurt accuracy? |
| D | TMG-GAN | All 41 | Does balancing alone help minority classes? |
| E | TMG-GAN | VAE-FSNID | Does our full method improve over B? |

Comparing D vs A → effect of balancing alone
Comparing E vs B → full proposed method vs original FSNID
Comparing E vs D → effect of VAE feature selection

---

## Hyperparameters

### Classifier (all configs)
| Parameter | Value |
|---|---|
| Hidden layers | [250, 250] |
| Activation | ReLU |
| Output | LogSoftmax |
| Optimizer | SGD |
| Learning rate | 0.01 |
| Momentum | 0.9 |
| Batch size | 1024 |
| Epochs | 50 (Phase 2) / 100 (Phase 1) |

### TMG-GAN
| Parameter | Value |
|---|---|
| Generator layers | [256, 256] |
| Discriminator layers | [256, 128] |
| Latent dimension | 100 |
| Optimizer | Adam (β1=0.5, β2=0.999) |
| Learning rate | 0.0002 |
| Epochs | 30 |
| Batch size | 256 |
| λ1 (classification) | 1.0 |
| λ2 (cosine similarity) | 0.5 |
| Target U2R count | 500 |
| Target R2L count | 2,000 |

### β-VAE
| Parameter | Value |
|---|---|
| Encoder layers | [128, 64] |
| Latent dimension | 10 |
| Decoder layers | [64, 128] |
| β (KL weight) | 4 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Epochs | 50 |
| Batch size | 256 |
| Null shuffle repeats | 10 |

---

## Dataset Statistics

| Split | Samples | Source |
|---|---|---|
| Train | 100,778 | 80% of KDDTrain+.txt |
| Test | 25,195 | 20% of KDDTrain+.txt |
| Split seed | 42 | numpy.random.seed(42) |

| Class | Label | Train | Test |
|---|---|---|---|
| Normal | 0 | 53,858 | 13,485 |
| DoS | 1 | 36,720 | 9,207 |
| Probe | 2 | 9,362 | 2,294 |
| R2L | 3 | 798 | 197 |
| U2R | 4 | 40 | 12 |
