"""
run_extension.py — Phase 2 Extension Pipeline
===============================================
Configs:
  A : No balancing   | All 41 features      (Phase 1 baseline)
  B : No balancing   | FSNID-MINE 12 feats  (Phase 1 result)
  D : TMG-GAN        | All 41 features
  E : TMG-GAN        | VAE-FSNID features   (Full proposed method)

Results are saved to results/extension_results.json after each config.
"""

import os, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

from src.classifier import FeatureClassifier
from src.tmg_gan    import TMG_GAN
from src.vae_fs     import vae_feature_selection

os.makedirs("results", exist_ok=True)
RESULTS_FILE = "results/extension_results.json"

# =========================================================
# SETTINGS
# =========================================================

BATCH_SIZE    = 1024
CLS_EPOCHS    = 50       # classifier epochs (50 is fast + accurate enough)
LEARNING_RATE = 0.01
NUM_CLASSES   = 5
SEEDS         = [0, 1, 2]   # 3 seeds — fast but statistically valid

FSNID_FEATURES   = [4, 11, 12, 15, 18, 20, 23, 26, 27, 31, 36, 38]
MINORITY_CLASSES = [3, 4]            # R2L, U2R
TMG_TARGET       = {3: 2000, 4: 500}
TMG_EPOCHS       = 30
TMG_BATCH        = 256
VAE_EPOCHS       = 50
CLASS_NAMES      = ["Normal", "DoS", "Probe", "R2L", "U2R"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =========================================================
# LOAD DATA
# =========================================================

x = np.load("data/processed/NSL-KDD/X_train.npy")
y = np.load("data/processed/NSL-KDD/y_train.npy", allow_pickle=True)

np.random.seed(42)
idx   = np.random.permutation(len(x))
split = int(0.8 * len(x))

X_train_raw = x[idx[:split]]
y_train_raw = y[idx[:split]]
X_test_raw  = x[idx[split:]]
y_test_raw  = y[idx[split:]]

print(f"Train: {X_train_raw.shape} | Test: {X_test_raw.shape}")

DOS   = {"back","land","neptune","pod","smurf","teardrop","apache2",
        "mailbomb","processtable","udpstorm","worm"}
PROBE = {"ipsweep","nmap","portsweep","satan","mscan","saint"}
R2L   = {"ftp_write","guess_passwd","imap","multihop","phf","spy",
        "warezclient","warezmaster","httptunnel","named","sendmail",
        "snmpgetattack","snmpguess","xlock","xsnoop","xterm"}
U2R   = {"buffer_overflow","loadmodule","perl","rootkit","ps","sqlattack"}

def mapCategory(label):
    if label == "normal": return 0
    if label in DOS:      return 1
    if label in PROBE:    return 2
    if label in R2L:      return 3
    if label in U2R:      return 4
    raise ValueError(f"Unknown: {label}")

y_train_enc = np.array([mapCategory(l) for l in y_train_raw])
y_test_enc  = np.array([mapCategory(l) for l in y_test_raw])

print("Train class counts:", np.bincount(y_train_enc))
print("Test  class counts:", np.bincount(y_test_enc))

# =========================================================
# HELPERS
# =========================================================

def run_experiment(X_tr, X_te, y_tr, y_te, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.long)
        ),
        batch_size=BATCH_SIZE, shuffle=True
    )

    model     = FeatureClassifier(X_tr.shape[1], NUM_CLASSES).to(device)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    criterion = nn.NLLLoss()

    model.train()
    for _ in range(CLS_EPOCHS):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            criterion(model(bx), by).backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = torch.argmax(
            model(torch.tensor(X_te, dtype=torch.float32).to(device)), dim=1
        ).cpu().numpy()

    acc    = accuracy_score(y_te, preds)
    f1_w   = f1_score(y_te, preds, average="weighted", zero_division=0)
    f1_cls = f1_score(y_te, preds, average=None, zero_division=0, labels=list(range(NUM_CLASSES)))
    return acc, f1_w, f1_cls.tolist()

def run_config(label, X_tr, X_te, y_tr, y_te):
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  Train: {X_tr.shape} | Features: {X_tr.shape[1]}")

    rows = []
    t0 = time.time()
    for seed in SEEDS:
        acc, f1_w, f1_cls = run_experiment(X_tr, X_te, y_tr, y_te, seed)
        rows.append({"seed": seed, "acc": acc, "f1_w": f1_w, "f1_cls": f1_cls})
        print(
            f"  Seed {seed}: Acc={acc*100:.2f}% | F1={f1_w*100:.2f}% | "
            f"R2L-F1={f1_cls[3]*100:.2f}% | U2R-F1={f1_cls[4]*100:.2f}%"
        )

    accs = np.array([r["acc"] for r in rows])
    f1s  = np.array([r["f1_w"] for r in rows])
    r2ls = np.array([r["f1_cls"][3] for r in rows])
    u2rs = np.array([r["f1_cls"][4] for r in rows])
    n    = len(SEEDS)

    result = {
        "label"       : label,
        "n_features"  : int(X_tr.shape[1]),
        "n_seeds"     : n,
        "cls_epochs"  : CLS_EPOCHS,
        "acc_mean"    : float(accs.mean()),
        "acc_ci"      : float(1.96 * accs.std(ddof=1) / np.sqrt(n)),
        "f1_mean"     : float(f1s.mean()),
        "f1_ci"       : float(1.96 * f1s.std(ddof=1) / np.sqrt(n)),
        "r2l_f1_mean" : float(r2ls.mean()),
        "u2r_f1_mean" : float(u2rs.mean()),
        "elapsed_min" : round((time.time() - t0) / 60, 2),
        "per_seed"    : rows
    }

    print(
        f"\n  RESULT: Acc={result['acc_mean']*100:.2f}% ±{result['acc_ci']*100:.2f}% | "
        f"F1={result['f1_mean']*100:.2f}% ±{result['f1_ci']*100:.2f}% | "
        f"R2L-F1={result['r2l_f1_mean']*100:.2f}% | "
        f"U2R-F1={result['u2r_f1_mean']*100:.2f}%"
    )
    
    return result

def save_results(all_results):
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  [Saved -> {RESULTS_FILE}]")

# =========================================================
# CONFIG A — Baseline
# =========================================================

result_A = run_config(
    "Config A | No balancing | All 41 features",
    X_train_raw, X_test_raw, y_train_enc, y_test_enc
)
all_results = {"A": result_A}
save_results(all_results)

# =========================================================
# CONFIG B — FSNID-MINE 12 features
# =========================================================

result_B = run_config(
    "Config B | No balancing | FSNID-MINE 12 features",
    X_train_raw[:, FSNID_FEATURES],
    X_test_raw[:,  FSNID_FEATURES],
    y_train_enc, y_test_enc
)
all_results["B"] = result_B
save_results(all_results)

# =========================================================
# CONFIG D — TMG-GAN + All 41 features
# =========================================================

print(f"\n{'='*55}")
print("  Config D | Training TMG-GAN...")
print(f"{'='*55}")

tmg = TMG_GAN(input_dim=41, num_classes=NUM_CLASSES, latent_dim=100)
tmg.train(
    X_train_raw, y_train_enc,
    minority_classes=MINORITY_CLASSES,
    epochs=TMG_EPOCHS,
    batch_size=TMG_BATCH
)

X_tr_D, y_tr_D = tmg.balance(
    X_train_raw, y_train_enc, target_counts=TMG_TARGET
)
print(f"  Balanced train size: {len(X_tr_D)} | "
    f"Class counts: {np.bincount(y_tr_D)}")

result_D = run_config(
    "Config D | TMG-GAN balancing | All 41 features",
    X_tr_D, X_test_raw, y_tr_D, y_test_enc
)
all_results["D"] = result_D
save_results(all_results)

# =========================================================
# CONFIG E — TMG-GAN + VAE-FSNID
# =========================================================

print(f"\n{'='*55}")
print("  Config E | Training VAE feature selector...")
print(f"{'='*55}")

vae_features, _, importances = vae_feature_selection(
    X_tr_D, input_dim=41, latent_dim=10, beta=4, vae_epochs=VAE_EPOCHS
)

if len(vae_features) == 0:
    print("  [WARNING] VAE-FS selected 0 features — using all 41.")
    vae_features = list(range(41))

print(f"\n  VAE-FS selected: {vae_features}")

result_E = run_config(
    f"Config E | TMG-GAN + VAE-FSNID ({len(vae_features)} features)",
    X_tr_D[:,    vae_features],
    X_test_raw[:, vae_features],
    y_tr_D, y_test_enc
)
all_results["E"] = result_E
all_results["vae_selected_features"] = vae_features
all_results["vae_importances"] = [float(x) for x in importances]
save_results(all_results)

# =========================================================
# FINAL TABLE
# =========================================================

print("\n\n" + "="*80)
print("  FINAL RESULTS")
print("="*80)
print(f"  {'Config':<45} {'Feats':>5} {'Accuracy':>14} {'W-F1':>14} "
     f"{'R2L-F1':>8} {'U2R-F1':>8}")
print("-"*80)

for key, r in all_results.items():
    if key in ["vae_selected_features", "vae_importances"]:
        continue
    print(
        f"  {r['label']:<45} "
        f"{r['n_features']:>5} "
        f"{r['acc_mean']*100:>7.2f}% ±{r['acc_ci']*100:.2f}%  "
        f"{r['f1_mean']*100:>7.2f}% ±{r['f1_ci']*100:.2f}%  "
        f"{r['r2l_f1_mean']*100:>6.2f}%  "
        f"{r['u2r_f1_mean']*100:>6.2f}%"
    )

print("="*80)
print(f"\nAll results saved to: {RESULTS_FILE}")
print(f"VAE selected features ({len(vae_features)}/41): {vae_features}")
print(f"Feature reduction: {(1-len(vae_features)/41)*100:.1f}%")