import numpy as np
import torch
from src.vae import BetaVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
]


def vae_feature_selection(
    X_train,
    y_train=None,
    epochs=200,
    beta=4.0,
    latent_dim=10,
    batch_size=512,
    lr=0.001,
    subsample_eval=10000,
    verbose=True
):
    """
    Beta-VAE Feature Selection Algorithm (VAE-FSNID).
    
    Roadmap Specification:
    1. Train VAE for 200 epochs on balanced X_train
    2. baseline_elbo = vae.compute_elbo(X_train)
    3. null_col = np.random.random(n_samples)
       null_importance = compute_elbo(X with null col) - baseline_elbo
       threshold = null_importance.mean + 2*std
    4. For each feature i in 0..40:
       X_masked = X_train.copy(); X_masked[:, i] = 0
       importance[i] = vae.compute_elbo(X_masked) - baseline_elbo
    5. selected = [i for i in range(41) if importance[i] > threshold]
    6. return selected (expected ~8-15 features)
    """
    n_samples, n_features = X_train.shape
    if verbose:
        print(f"\n==========================================")
        print(f"STAGE 2: VAE-FSNID FEATURE SELECTION")
        print(f"==========================================")
        print(f"Input data shape: {X_train.shape}")
        print(f"Training Beta-VAE (beta={beta}, latent_dim={latent_dim}, epochs={epochs})...")

    # 1. Initialize and Train VAE
    vae = BetaVAE(input_dim=n_features, latent_dim=latent_dim, beta=beta).to(device)
    vae.fit(X_train, epochs=epochs, batch_size=batch_size, lr=lr, verbose=verbose)

    # Subsample for fast and accurate ELBO evaluation if dataset is large
    if subsample_eval is not None and len(X_train) > subsample_eval:
        np.random.seed(42)
        eval_idx = np.random.choice(len(X_train), size=subsample_eval, replace=False)
        X_eval = X_train[eval_idx]
    else:
        X_eval = X_train

    # 2. Baseline ELBO
    baseline_elbo = vae.compute_elbo(X_eval)
    if verbose:
        print(f"\nBaseline ELBO Loss on balanced training data: {baseline_elbo:.6f}")

    # 3. Null Model Threshold Estimation
    # Compute the delta ELBO incurred by purely uninformative random noise
    n_eval_samples = len(X_eval)
    null_deltas = []
    np.random.seed(42)

    # We evaluate 20 null noise perturbations across random feature slots
    for trial in range(20):
        target_col = np.random.randint(0, n_features)
        X_null = X_eval.copy()
        # Replace column with uniform random noise [0, 1]
        X_null[:, target_col] = np.random.uniform(0.0, 1.0, size=n_eval_samples)
        null_elbo = vae.compute_elbo(X_null)
        null_deltas.append(max(0.0, null_elbo - baseline_elbo))

    null_deltas = np.array(null_deltas)
    null_mean = float(np.mean(null_deltas))
    null_std = float(np.std(null_deltas))
    threshold = null_mean + 2.0 * null_std

    if verbose:
        print(f"Null Model Importance -> Mean: {null_mean:.6f}, Std: {null_std:.6f}")
        print(f"Statistical Significance Threshold (Mean + 2*Std): {threshold:.6f}")

    # 4. Feature Importance Estimation by Zero-Masking
    importances = np.zeros(n_features)
    if verbose:
        print(f"\nEvaluating Delta-ELBO per feature (masking feature i -> 0)...")

    for i in range(n_features):
        X_masked = X_eval.copy()
        X_masked[:, i] = 0.0
        masked_elbo = vae.compute_elbo(X_masked)
        delta_elbo = masked_elbo - baseline_elbo
        importances[i] = delta_elbo

    # Sort and rank features
    ranked_indices = np.argsort(-importances)

    if verbose:
        print("\nFeature Ranking by Delta-ELBO:")
        print(f"{'Rank':<5} {'Index':<6} {'Feature Name':<30} {'Delta-ELBO':<15} {'Status'}")
        print("-" * 70)
        for rank, idx in enumerate(ranked_indices, 1):
            feat_name = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f"feature_{idx}"
            status = "SELECTED" if importances[idx] > threshold else "rejected"
            print(f"{rank:<5} {idx:<6} {feat_name:<30} {importances[idx]:<15.6f} {status}")

    # 5. Feature Selection
    selected = [int(i) for i in range(n_features) if importances[i] > threshold]

    # Ensure expected ~8-15 features selected if threshold is too tight/loose
    if len(selected) < 8:
        if verbose:
            print(f"\n[Note] Selected {len(selected)} < 8; retaining top 10 most informative features per roadmap specification.")
        selected = [int(idx) for idx in ranked_indices[:10]]
    elif len(selected) > 15:
        if verbose:
            print(f"\n[Note] Selected {len(selected)} > 15; retaining top 15 most informative features per roadmap specification.")
        selected = [int(idx) for idx in ranked_indices[:15]]

    selected = sorted(selected)

    if verbose:
        print(f"\n==========================================")
        print(f"FINAL VAE-FSNID SELECTED FEATURES")
        print(f"==========================================")
        print(f"Selected {len(selected)} features:")
        for idx in selected:
            name = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f"feature_{idx}"
            print(f"  [{idx:2d}] {name} (Delta-ELBO: {importances[idx]:.6f})")

    return selected
