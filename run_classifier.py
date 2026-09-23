import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from src.fsnid_fs import fsnid_selection

from src.classifier import FeatureClassifier

# =========================================================
# SETTINGS
# =========================================================

BATCH_SIZE = 1024
EPOCHS = 100
LEARNING_RATE = 0.01

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

# =========================================================
# LOAD DATA
# =========================================================
x = np.load("data/processed/NSL-KDD/X_train.npy")
y = np.load("data/processed/NSL-KDD/y_train.npy", allow_pickle=True)

np.random.seed(42)

indices = np.random.permutation(len(x))
split = int(0.8*len(x))
train_index, test_index = indices[:split], indices[split:]

X_train = x[train_index]
y_train = y[train_index]

X_test = x[test_index]
y_test = y[test_index]

print("Train:", X_train.shape)
print("Test :", X_test.shape)

# =========================================================
# LABEL ENCODING
# =========================================================

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

y_train_encoded = np.array([mapCategory(i) for i in y_train])
y_test_encoded = np.array([mapCategory(i) for i in y_test])

num_classes = 5

print("Number of classes: ", num_classes)
print("Training class counts: ", np.bincount(y_train_encoded))
print("Testing class counts: ", np.bincount(y_test_encoded))

# =========================================================
# TRAIN + EVALUATE
# =========================================================

def run_experiment(X_train_data, X_test_data, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_X = torch.tensor(
        X_train_data,
        dtype=torch.float32
    )

    train_y = torch.tensor(
        y_train_encoded,
        dtype=torch.long
    )

    test_X = torch.tensor(
        X_test_data,
        dtype=torch.float32
    )

    train_dataset = TensorDataset(
        train_X,
        train_y
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = FeatureClassifier(
        input_size=X_train_data.shape[1],
        num_classes=num_classes
    ).to(device)

    criterion = nn.NLLLoss()

    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150,250], gamma=0.1)
    
    model.train()

    for epoch in range(EPOCHS):

        total_loss = 0

        for batch_X, batch_y in train_loader:

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            predictions = model(batch_X)

            loss = criterion(
                predictions,
                batch_y
            )

            loss.backward()

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:

            print(
                f"Epoch {epoch + 1}/{EPOCHS} "
                f"Loss: {total_loss / len(train_loader):.4f}"
            )


    # =====================================================
    # TEST
    # =====================================================

    model.eval()

    with torch.no_grad():

        predictions = model(
            test_X.to(device)
        )

        predicted_labels = torch.argmax(
            predictions,
            dim=1
        ).cpu().numpy()


    accuracy = accuracy_score(
        y_test_encoded,
        predicted_labels
    )

    f1 = f1_score(
        y_test_encoded,
        predicted_labels,
        average="weighted",
        zero_division=0
    )

    return accuracy, f1

def conclusion(results):
    results = np.array(results)

    mean = results.mean(axis=0)
    std = results.std(axis=0, ddof=1)

    ci = 1.96*std/np.sqrt(len(results))
    return mean, ci

# =========================================================
# BASELINE — ALL 41 FEATURES
# =========================================================
seeds = [0, 1, 2]
total = []

print("\n================================")
print("BASELINE — ALL 41 FEATURES")
print("================================")
for i in seeds:
    print(f"\nSEED {i}")

    acc, f1 = run_experiment(X_train, X_test, i)
    total.append([acc, f1])

    print(
        f"Accuracy: {acc:.4f} | "
        f"F1: {f1:.4f}"
    )

totalMean, totalCI = conclusion(total)

print("\nBASELINE RESULTS")
print(
    f"Accuracy: "
    f"{totalMean[0] * 100:.2f}% "
    f"± {totalCI[0] * 100:.2f}%"
)

print(
    f"Weighted F1: "
    f"{totalMean[1] * 100:.2f}% "
    f"± {totalCI[1] * 100:.2f}%"
)

# =========================================================
# FSNID FEATURE SELECTION
# =========================================================
FSNID_FEATURES = [4, 11, 12, 15, 18, 20, 23, 26, 27, 31, 36, 38]
print("FSNID Selected Features:", FSNID_FEATURES)

# =========================================================
# FINAL EXPERIMENT — 5 SEEDS
# =========================================================
seeds = [0, 1, 2, 3, 4]

X_train_fsnid = X_train[:, FSNID_FEATURES]
X_test_fsnid = X_test[:, FSNID_FEATURES]

total_fsnid = []

for i in seeds:
    print("\n================================")
    print(f"FSNID SEED {i}")
    print("================================")

    acc, f1 = run_experiment(X_train_fsnid, X_test_fsnid, i)
    total_fsnid.append([acc, f1])

    print(
        f"Accuracy: {acc:.4f} | "
        f"F1: {f1:.4f}"
    )

# =========================================================
# MEAN + 95% CONFIDENCE INTERVAL
# =========================================================
totalMean, totalCI = conclusion(total)
fsnid_mean, fsnid_ci = conclusion(total_fsnid)

print("\n\n================================")
print("FINAL RESULTS")
print("================================")

print("\nALL 41 FEATURES")
print(
    f"Accuracy: "
    f"{totalMean[0] * 100:.2f}% "
    f"± {totalCI[0] * 100:.2f}%"
)

print(
    f"Weighted F1: "
    f"{totalMean[1] * 100:.2f}% "
    f"± {totalCI[1] * 100:.2f}%"
)


print("\nFSNID SELECTED FEATURES")
print(
    f"Accuracy: "
    f"{fsnid_mean[0] * 100:.2f}% "
    f"± {fsnid_ci[0] * 100:.2f}%"
)

print(
    f"Weighted F1: "
    f"{fsnid_mean[1] * 100:.2f}% "
    f"± {fsnid_ci[1] * 100:.2f}%"
)