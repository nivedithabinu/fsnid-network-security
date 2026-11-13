import numpy as np
from src.vae import train_vae

def vae_feature_selection(X_train, input_dim=41, latent_dim=10, beta=4, vae_epochs=200, n_null=10):
    """
    Select features using β-VAE reconstruction sensitivity.

    Algorithm
    ---------
    1. Train a β-VAE on X_train.
    2. Compute baseline ELBO on the full dataset.
    3. Estimate null threshold: shuffle each of n_null random features and measure ΔELBO. Threshold = mean + 2×std of null ΔELBOs.
    4. For each feature i: shuffle column i, measure ΔELBO.
    5. Keep feature i if its ΔELBO > threshold.

    Shuffling is used instead of zeroing to preserve the marginal distribution of each feature while destroying its conditional relationship with others (permutation importance).

    Parameters
    ----------
    X_train    : np.ndarray (n, input_dim)
    input_dim  : number of features
    latent_dim : VAE latent dimension
    beta       : β for KL weight in ELBO
    vae_epochs : epochs to train the VAE
    n_null     : number of random-feature shuffles for null threshold

    Returns
    -------
    selected   : list of selected feature indices
    vae        : trained BetaVAE model
    importances: list of ΔELBO scores for all features
    """

    # ── Step 1: Train VAE ─────────────────────────────────────
    vae = train_vae(
        X_train,
        input_dim=input_dim,
        latent_dim=latent_dim,
        beta=beta,
        epochs=vae_epochs
    )

    # ── Step 2: Baseline ELBO ─────────────────────────────────
    baseline_elbo = vae.compute_elbo(X_train)
    print(f"\nBaseline ELBO : {baseline_elbo:.6f}")

    # ── Step 3: Null threshold ────────────────────────────────
    null_deltas = []
    n = X_train.shape[0]

    for _ in range(n_null):
        X_null = X_train.copy()
        rand_feat = np.random.randint(0, input_dim)
        X_null[:, rand_feat] = X_train[
            np.random.permutation(n), rand_feat
        ]
        null_elbo = vae.compute_elbo(X_null)
        null_deltas.append(null_elbo - baseline_elbo)

    null_deltas = np.array(null_deltas)
    threshold   = null_deltas.mean() + 2.0 * null_deltas.std()

    print(f"Null threshold: {threshold:.6f} "
        f"(mean={null_deltas.mean():.6f}, std={null_deltas.std():.6f})")

    # ── Step 4: Score every feature ───────────────────────────
    importances = []

    print("\nScoring features (permutation sensitivity)...")
    for i in range(input_dim):
        X_masked = X_train.copy()
        # Shuffle column i (permutation importance)
        X_masked[:, i] = X_train[np.random.permutation(n), i]
        masked_elbo = vae.compute_elbo(X_masked)
        delta = masked_elbo - baseline_elbo
        importances.append(delta)

    # ── Step 5: Select ────────────────────────────────────────
    selected = [
        i for i, d in enumerate(importances)
        if d > threshold
    ]

    print("\n--- VAE Feature Selection Results ---")
    for i, d in enumerate(importances):
        status = "KEEP" if d > threshold else "DROP"
        print(
            f"  Feature {i:2d}: ΔELBO = {d:+.6f} | "
            f"threshold = {threshold:.6f} | {status}"
        )

    print(f"\nVAE-FS selected {len(selected)}/{input_dim} features: {selected}")
    print(
        f"Dimensionality reduction: "
        f"{(1 - len(selected)/input_dim)*100:.1f}%"
    )

    return selected, vae, importances