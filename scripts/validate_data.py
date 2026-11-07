import numpy as np
from collections import Counter

# PERFORMING DATA VALIDATION - Reuse the existing classifier's split and class mapping

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "NSL-KDD"

# 1. Load the processed data
X = np.load(f"{DATA_DIR}/X_train.npy")
y = np.load(f"{DATA_DIR}/y_train.npy", allow_pickle=True)

print("Original dataset shape: ", X.shape)
print("Original labels: ", len(np.unique(y)))

# 2. Five-Class Attack Mapping
DOS_ATTACKS = {
    "back", "land", "neptune", "pod", "smurf", "teardrop",
    "apache2", "mailbomb", "processtable", "udpstorm", "worm"
}

PROBE_ATTACKS = {
    "ipsweep", "nmap", "portsweep", "satan", "mscan", "saint"
}

R2L_ATTACKS = {
    "ftp_write", "guess_passwd", "imap", "multihop", "phf",
    "spy", "warezclient", "warezmaster", "httptunnel",
    "named", "sendmail", "snmpgetattack", "snmpguess",
    "xlock", "xsnoop", "xterm"
}

U2R_ATTACKS = {
    "buffer_overflow", "loadmodule", "perl",
    "rootkit", "ps", "sqlattack"
}

def map_category(label):
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
        raise ValueError(f"Unmapped attack label: {label}")

CLASS_NAMES = {0: "Normal", 1: "DoS", 2: "Probe", 3: "R2L", 4: "U2R"}

# 3. Map all class labels

y_mapped = np.array([map_category(i) for i in y], dtype=np.int64)

# 4. Reproducing the existing 80:20 split
np.random.seed(42)

indices = np.random.permutation(len(X))
split = int(0.8*len(X))

train_indices = indices[:split]
test_indices = indices[split:]

X_train = X[train_indices]
X_test = X[test_indices]

y_train = y_mapped[train_indices]
y_test = y_mapped[test_indices]

# 5. Display Class Distributions
def display_distribution(title, labels):
    counts = Counter(labels)

    print(f"\n{title}")
    print("-" * 42)

    for i, j in CLASS_NAMES.items():
        print(
            f"{j:<10}: "
            f"{counts.get(i, 0):>8}"
        )

    print("-" * 42)
    print(f"{'Total':<10}: {len(labels):>8}")


display_distribution("TRAINING DISTRIBUTION", y_train)
display_distribution("TEST DISTRIBUTION", y_test)

# 6. Final Validation
assert X_train.shape[0] == len(y_train)
assert X_test.shape[0] == len(y_test)

assert set(np.unique(y_train)) == {0, 1, 2, 3, 4}
assert set(np.unique(y_test)) == {0, 1, 2, 3, 4}

print("\nAll five classes are present.")
print("Data validation completed successfully.")