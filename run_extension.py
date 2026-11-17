import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from src.classifier import FeatureClassifier
from src.tmg_gan import TMG_GAN
from src.vae_fs import vae_feature_selection, FEATURE_NAMES

# =========================================================
# CONFIGURATION & HYPERPARAMETERS
# =========================================================
BATCH_SIZE = 1024
EPOCHS = 100
LEARNING_RATE = 0.01
SEEDS = [0, 1, 2, 3, 4]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FSNID_MINE_FEATURES = [4, 11, 12, 15, 18, 20, 23, 26, 27, 31, 36, 38]
CLASS_NAMES = ["Normal", "DoS", "Probe", "R2L", "U2R"]

# Attack labels mapping
DOS_ATTACKS = {
    "back", "land", "neptune", "pod", "smurf",
    "teardrop", "apache2", "mailbomb", "processtable",
    "udpstorm", "worm"
}
PROBE_ATTACKS = {
    "ipsweep", "nmap", "portsweep", "satan",
    "mscan", "saint"
}
R2L_ATTACKS = {
    "ftp_write", "guess_passwd", "imap", "multihop",
    "phf", "spy", "warezclient", "warezmaster",
    "httptunnel", "named", "sendmail", "snmpgetattack",
    "snmpguess", "xlock", "xsnoop", "xterm"
}
U2R_ATTACKS = {
    "buffer_overflow", "loadmodule", "perl", "rootkit",
    "ps", "sqlattack"
}

def mapCategory(label):
    if label == "normal":
        return 0
    elif label in DOS_ATTACKS:
        return 1
    elif label in PROBE_ATTACKS:
        return 2
    elif label in R2L_ATTACKS:
        return 3
    elif label in U2R_ATTACKS:
        return 4
    else:
        raise ValueError(f"Unknown attack label: {label}")


# =========================================================
# 1. DATA PREPARATION (80:20 split, seed=42)
# =========================================================
def load_data():
    x = np.load("data/processed/NSL-KDD/X_train.npy")
    y = np.load("data/processed/NSL-KDD/y_train.npy", allow_pickle=True)

    np.random.seed(42)
    indices = np.random.permutation(len(x))
    split = int(0.8 * len(x))
    train_index, test_index = indices[:split], indices[split:]

    X_train = x[train_index]
    y_train = np.array([mapCategory(i) for i in y[train_index]])

    # X_test is NEVER touched during balancing or feature selection!
    X_test = x[test_index]
    y_test = np.array([mapCategory(i) for i in y[test_index]])

    print(f"[Data] Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")
    print(f"[Data] Testing samples:  {X_test.shape[0]}, Features: {X_test.shape[1]}")
    print(f"[Data] Training class counts: {np.bincount(y_train)}")
    print(f"[Data] Testing class counts:  {np.bincount(y_test)}")

    return X_train, y_train, X_test, y_test


# =========================================================
# 2. BALANCING PIPELINES
# =========================================================
def get_balanced_data_ctgan(X_train, y_train, target_counts={4: 500, 3: 2000}):
    cache_path_x = "data/processed/NSL-KDD/X_train_ctgan.npy"
    cache_path_y = "data/processed/NSL-KDD/y_train_ctgan.npy"

    if os.path.exists(cache_path_x) and os.path.exists(cache_path_y):
        print(f"[CTGAN] Loading cached balanced dataset from {cache_path_x}...")
        return np.load(cache_path_x), np.load(cache_path_y)

    print("\n[CTGAN] Generating synthetic samples for minority classes...")
    from ctgan import CTGAN

    cols = [f"f_{i}" for i in range(X_train.shape[1])]
    synthetic_X = [X_train]
    synthetic_y = [y_train]

    for cls_id, target_n in target_counts.items():
        curr_n = np.sum(y_train == cls_id)
        needed = target_n - curr_n
        if needed > 0:
            print(f"[CTGAN] Training CTGAN on class {cls_id} ({CLASS_NAMES[cls_id]}: {curr_n} samples) -> Generating {needed} samples...")
            cls_df = pd.DataFrame(X_train[y_train == cls_id], columns=cols)
            batch_sz = min(50, len(cls_df))
            ctgan = CTGAN(epochs=30, batch_size=batch_sz, verbose=False)
            ctgan.fit(cls_df)
            synth_df = ctgan.sample(needed)
            synth_x = np.clip(synth_df.to_numpy(), 0.0, 1.0)
            synth_y = np.full(needed, cls_id, dtype=y_train.dtype)
            synthetic_X.append(synth_x)
            synthetic_y.append(synth_y)

    X_bal = np.concatenate(synthetic_X, axis=0)
    y_bal = np.concatenate(synthetic_y, axis=0)
    shuffle_idx = np.random.permutation(len(X_bal))
    X_bal = X_bal[shuffle_idx]
    y_bal = y_bal[shuffle_idx]

    np.save(cache_path_x, X_bal)
    np.save(cache_path_y, y_bal)
    print(f"[CTGAN] Saved balanced dataset to {cache_path_x}. Class counts: {np.bincount(y_bal)}")
    return X_bal, y_bal


def get_balanced_data_tmggan(X_train, y_train, target_counts={4: 500, 3: 2000}, epochs=60):
    cache_path_x = "data/processed/NSL-KDD/X_train_tmggan.npy"
    cache_path_y = "data/processed/NSL-KDD/y_train_tmggan.npy"

    if os.path.exists(cache_path_x) and os.path.exists(cache_path_y):
        print(f"[TMG-GAN] Loading cached balanced dataset from {cache_path_x}...")
        return np.load(cache_path_x), np.load(cache_path_y)

    print("\n[TMG-GAN] Training TMG-GAN on training set...")
    tmggan = TMG_GAN(input_dim=41, num_classes=5, batch_size=256)
    tmggan.train(X_train, y_train, epochs=epochs, verbose=True)
    X_bal, y_bal = tmggan.balance(X_train, y_train, target_counts=target_counts)

    np.save(cache_path_x, X_bal)
    np.save(cache_path_y, y_bal)
    print(f"[TMG-GAN] Saved balanced dataset to {cache_path_x}. Class counts: {np.bincount(y_bal)}")
    return X_bal, y_bal


# =========================================================
# 3. FEATURE SELECTION PIPELINE
# =========================================================
def get_vae_selected_features(X_balanced, y_balanced, epochs=100):
    cache_path = "data/processed/NSL-KDD/vae_selected_features.npy"
    if os.path.exists(cache_path):
        feats = np.load(cache_path).tolist()
        print(f"[VAE-FS] Loaded cached selected features: {feats}")
        return feats

    print("\n[VAE-FS] Running Beta-VAE Feature Selection...")
    feats = vae_feature_selection(X_balanced, y_balanced, epochs=epochs, verbose=True)
    np.save(cache_path, np.array(feats))
    return feats


# =========================================================
# 4. CLASSIFICATION & EVALUATION ENGINE
# =========================================================
def train_and_evaluate(X_train_data, y_train_data, X_test_data, y_test_data, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_X = torch.tensor(X_train_data, dtype=torch.float32)
    train_y = torch.tensor(y_train_data, dtype=torch.long)
    test_X = torch.tensor(X_test_data, dtype=torch.float32)

    dataset = TensorDataset(train_X, train_y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    input_dim = X_train_data.shape[1]
    model = FeatureClassifier(input_size=input_dim, num_classes=5).to(DEVICE)

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)

    model.train()
    for epoch in range(EPOCHS):
        for bx, by in loader:
            bx = bx.to(DEVICE)
            by = by.to(DEVICE)
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

    # Evaluation on X_test
    model.eval()
    with torch.no_grad():
        preds = model(test_X.to(DEVICE))
        pred_labels = torch.argmax(preds, dim=1).cpu().numpy()

    acc = accuracy_score(y_test_data, pred_labels)
    wf1 = f1_score(y_test_data, pred_labels, average="weighted", zero_division=0)
    mf1 = f1_score(y_test_data, pred_labels, average="macro", zero_division=0)

    # Per-class F1
    per_class_f1 = f1_score(y_test_data, pred_labels, average=None, zero_division=0)
    cm = confusion_matrix(y_test_data, pred_labels, labels=[0, 1, 2, 3, 4])

    return {
        "accuracy": acc,
        "weighted_f1": wf1,
        "macro_f1": mf1,
        "normal_f1": per_class_f1[0],
        "dos_f1": per_class_f1[1],
        "probe_f1": per_class_f1[2],
        "r2l_f1": per_class_f1[3],
        "u2r_f1": per_class_f1[4],
        "confusion_matrix": cm
    }


def compute_confidence_intervals(results_list):
    """
    Computes Mean +/- 95% Confidence Interval (1.96 * std / sqrt(N)).
    """
    keys = ["accuracy", "weighted_f1", "macro_f1", "u2r_f1", "r2l_f1", "probe_f1", "dos_f1", "normal_f1"]
    summary = {}
    n = len(results_list)

    for k in keys:
        vals = [r[k] for r in results_list]
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        ci = float(1.96 * std / np.sqrt(n)) if n > 1 else 0.0
        summary[k] = {"mean": mean, "ci": ci, "std": std}

    return summary


# =========================================================
# 5. MAIN EXPERIMENT RUNNER (ALL 5 CONFIGS)
# =========================================================
def run_all_configurations():
    print("=" * 80)
    print("TMG-GAN + VAE-FSNID: 5-CONFIGURATION ABLATION STUDY")
    print("=" * 80)

    # Step 1: Load Data
    X_train_raw, y_train_raw, X_test_raw, y_test = load_data()

    # Step 2: Prepare Balancers
    print("\n--- PREPARING BALANCED DATASETS ---")
    X_train_ctgan, y_train_ctgan = get_balanced_data_ctgan(X_train_raw, y_train_raw)
    X_train_tmggan, y_train_tmggan = get_balanced_data_tmggan(X_train_raw, y_train_raw, epochs=60)

    # Step 3: Run Feature Selection
    print("\n--- RUNNING VAE FEATURE SELECTION ---")
    vae_selected = get_vae_selected_features(X_train_tmggan, y_train_tmggan, epochs=100)

    # Define Configurations
    configs = [
        {
            "id": "A",
            "name": "Baseline (No balancing, all 41 features)",
            "short": "Baseline",
            "X_train": X_train_raw,
            "y_train": y_train_raw,
            "features": list(range(41)),
            "purpose": "Full-feature baseline"
        },
        {
            "id": "B",
            "name": "FSNID-MINE (No balancing, 12 MINE features)",
            "short": "FSNID-MINE",
            "X_train": X_train_raw,
            "y_train": y_train_raw,
            "features": FSNID_MINE_FEATURES,
            "purpose": "Mid-sem replication"
        },
        {
            "id": "C",
            "name": "CTGAN + All 41 (CTGAN balancing, all 41 features)",
            "short": "CTGAN + All 41",
            "X_train": X_train_ctgan,
            "y_train": y_train_ctgan,
            "features": list(range(41)),
            "purpose": "Balancing effect only"
        },
        {
            "id": "D",
            "name": "TMG-GAN + All 41 (TMG-GAN balancing, all 41 features)",
            "short": "TMG-GAN + All 41",
            "X_train": X_train_tmggan,
            "y_train": y_train_tmggan,
            "features": list(range(41)),
            "purpose": "TMG-GAN vs CTGAN"
        },
        {
            "id": "E",
            "name": "TMG-GAN + VAE-FSNID (Full proposed 3-stage method)",
            "short": "TMG-GAN + VAE-FSNID",
            "X_train": X_train_tmggan,
            "y_train": y_train_tmggan,
            "features": vae_selected,
            "purpose": "Full proposed method"
        }
    ]

    all_results = {}
    last_confusion_matrices = {}

    for cfg in configs:
        cfg_id = cfg["id"]
        short_name = cfg["short"]
        feat_subset = cfg["features"]
        n_feats = len(feat_subset)

        print(f"\n{'='*70}")
        print(f"RUNNING CONFIG {cfg_id}: {cfg['name']}")
        print(f"Features in use ({n_feats}): {feat_subset}")
        print(f"{'='*70}")

        X_tr = cfg["X_train"][:, feat_subset]
        y_tr = cfg["y_train"]
        X_te = X_test_raw[:, feat_subset]

        seed_results = []
        for s in SEEDS:
            print(f"  [Seed {s}] Training FeatureClassifier...", end=" ")
            res = train_and_evaluate(X_tr, y_tr, X_te, y_test, seed=s)
            seed_results.append(res)
            print(f"Acc: {res['accuracy']*100:.2f}% | W-F1: {res['weighted_f1']*100:.2f}% | U2R F1: {res['u2r_f1']*100:.2f}% | R2L F1: {res['r2l_f1']*100:.2f}%")

        summary = compute_confidence_intervals(seed_results)
        summary["config_id"] = cfg_id
        summary["method"] = short_name
        summary["num_features"] = n_feats
        summary["features"] = feat_subset
        summary["purpose"] = cfg["purpose"]
        all_results[cfg_id] = summary

        # Average confusion matrix
        cms = np.array([r["confusion_matrix"] for r in seed_results])
        last_confusion_matrices[cfg_id] = np.round(np.mean(cms, axis=0)).astype(int)

    # =========================================================
    # 6. RESULTS PRESENTATION & SAVING
    # =========================================================
    print("\n\n" + "=" * 95)
    print("FINAL ABLATION STUDY RESULTS (5 SEEDS, 95% CONFIDENCE INTERVALS)")
    print("=" * 95)
    header = f"{'Config':<8} {'Method':<22} {'Features':<10} {'Accuracy':<18} {'Weighted F1':<18} {'U2R F1':<12} {'R2L F1':<12}"
    print(header)
    print("-" * 95)

    rows = []
    for cfg_id in ["A", "B", "C", "D", "E"]:
        res = all_results[cfg_id]
        acc_str = f"{res['accuracy']['mean']*100:.2f}% ± {res['accuracy']['ci']*100:.2f}%"
        wf1_str = f"{res['weighted_f1']['mean']*100:.2f}% ± {res['weighted_f1']['ci']*100:.2f}%"
        u2r_str = f"{res['u2r_f1']['mean']*100:.2f}% ± {res['u2r_f1']['ci']*100:.2f}%"
        r2l_str = f"{res['r2l_f1']['mean']*100:.2f}% ± {res['r2l_f1']['ci']*100:.2f}%"
        print(f"{cfg_id:<8} {res['method']:<22} {res['num_features']:<10} {acc_str:<18} {wf1_str:<18} {u2r_str:<12} {r2l_str:<12}")

        rows.append({
            "Config": cfg_id,
            "Method": res["method"],
            "Features": res["num_features"],
            "Accuracy_Mean": res["accuracy"]["mean"],
            "Accuracy_CI": res["accuracy"]["ci"],
            "Weighted_F1_Mean": res["weighted_f1"]["mean"],
            "Weighted_F1_CI": res["weighted_f1"]["ci"],
            "U2R_F1_Mean": res["u2r_f1"]["mean"],
            "U2R_F1_CI": res["u2r_f1"]["ci"],
            "R2L_F1_Mean": res["r2l_f1"]["mean"],
            "R2L_F1_CI": res["r2l_f1"]["ci"],
            "Macro_F1_Mean": res["macro_f1"]["mean"],
            "Macro_F1_CI": res["macro_f1"]["ci"]
        })

    print("=" * 95)

    # Save to experiments directory
    os.makedirs("experiments", exist_ok=True)
    df_results = pd.DataFrame(rows)
    df_results.to_csv("experiments/ablation_results.csv", index=False)

    with open("experiments/ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # =========================================================
    # 7. GENERATE PUBLICATION-QUALITY PLOTS
    # =========================================================
    generate_plots(all_results, last_confusion_matrices, vae_selected)
    print("\n[Plots] All comparison plots and confusion matrices saved to experiments/ directory.")


def generate_plots(all_results, confusion_matrices, vae_selected):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    configs = ["A", "B", "C", "D", "E"]
    labels = [all_results[c]["method"] for c in configs]

    accs = [all_results[c]["accuracy"]["mean"] * 100 for c in configs]
    acc_errs = [all_results[c]["accuracy"]["ci"] * 100 for c in configs]

    wf1s = [all_results[c]["weighted_f1"]["mean"] * 100 for c in configs]
    wf1_errs = [all_results[c]["weighted_f1"]["ci"] * 100 for c in configs]

    u2r = [all_results[c]["u2r_f1"]["mean"] * 100 for c in configs]
    u2r_errs = [all_results[c]["u2r_f1"]["ci"] * 100 for c in configs]

    r2l = [all_results[c]["r2l_f1"]["mean"] * 100 for c in configs]
    r2l_errs = [all_results[c]["r2l_f1"]["ci"] * 100 for c in configs]

    # Plot 1: 4-Panel Ablation Comparison Bar Chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ["#4A90E2", "#7ED321", "#F5A623", "#BD10E0", "#D0021B"]

    # (a) Overall Accuracy
    axes[0, 0].bar(labels, accs, yerr=acc_errs, capsize=5, color=colors, alpha=0.85, edgecolor="black")
    axes[0, 0].set_title("(a) Overall Accuracy (%)", fontsize=13, fontweight="bold")
    axes[0, 0].set_ylim(80, 100)
    axes[0, 0].tick_params(axis="x", rotation=20)

    # (b) Weighted F1
    axes[0, 1].bar(labels, wf1s, yerr=wf1_errs, capsize=5, color=colors, alpha=0.85, edgecolor="black")
    axes[0, 1].set_title("(b) Weighted F1-Score (%)", fontsize=13, fontweight="bold")
    axes[0, 1].set_ylim(80, 100)
    axes[0, 1].tick_params(axis="x", rotation=20)

    # (c) U2R F1 (Minority Attack - 40 samples in train)
    axes[1, 0].bar(labels, u2r, yerr=u2r_errs, capsize=5, color=colors, alpha=0.85, edgecolor="black")
    axes[1, 0].set_title("(c) U2R Attack F1-Score (%) [Rare Attack]", fontsize=13, fontweight="bold")
    axes[1, 0].tick_params(axis="x", rotation=20)

    # (d) R2L F1 (Minority Attack - 798 samples in train)
    axes[1, 1].bar(labels, r2l, yerr=r2l_errs, capsize=5, color=colors, alpha=0.85, edgecolor="black")
    axes[1, 1].set_title("(d) R2L Attack F1-Score (%) [Rare Attack]", fontsize=13, fontweight="bold")
    axes[1, 1].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.savefig("experiments/ablation_comparison.png", dpi=300)
    plt.close()

    # Plot 2: Confusion Matrices for Config A, B, D, E
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    plot_cfgs = [("A", axes[0, 0]), ("B", axes[0, 1]), ("D", axes[1, 0]), ("E", axes[1, 1])]

    for cfg_id, ax in plot_cfgs:
        cm = confusion_matrices[cfg_id]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        ax.set_title(f"Config {cfg_id}: {all_results[cfg_id]['method']}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    plt.tight_layout()
    plt.savefig("experiments/confusion_matrices.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    run_all_configurations()
