import numpy as np

from src.fsnid_fs import fsnid_selection


# =========================================================
# FEATURE NAMES
# =========================================================

FEATURE_NAMES = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate"
]


# =========================================================
# 1. LOAD PROCESSED NSL-KDD
# =========================================================

X_train = np.load(
    "data/processed/NSL-KDD/X_train.npy"
)

y_train = np.load(
    "data/processed/NSL-KDD/y_train.npy",
    allow_pickle=True
)

print("Full X_train shape:", X_train.shape)
print("Full y_train shape:", y_train.shape)


# =========================================================
# 2. SMALL SUBSET FOR TESTING
# =========================================================
np.random.seed(42)

indices = np.random.choice(
    len(X_train),
    size=5000,
    replace=False
)

X = X_train[indices]
y = y_train[indices]
print("Test X shape:", X.shape)
print("Test y shape:", y.shape)


# =========================================================
# 3. MULTICLASS LABEL ENCODING
# =========================================================

# IMPORTANT:
# Create the mapping using ONLY the labels present
# in this subset.
#
# This guarantees labels are:
# 0, 1, 2, ..., num_classes-1

unique_labels = sorted(np.unique(y))

label_to_id = {
    label: index
    for index, label in enumerate(unique_labels)
}

y_encoded = np.array(
    [label_to_id[label] for label in y],
    dtype=np.int64
)

print("\nNumber of classes:", len(unique_labels))
print("Encoded classes:", np.unique(y_encoded))

print("\nLabel mapping:")
for label, index in label_to_id.items():
    print(f"{index}: {label}")


# =========================================================
# 4. RUN FSNID
# =========================================================

selector = fsnid_selection(
    features=X,
    targets=y_encoded,
    num_iterations=100
)

selected_features = selector.run_main()


# =========================================================
# 5. CONVERT INDICES TO FEATURE NAMES
# =========================================================

selected_feature_names = [
    FEATURE_NAMES[i]
    for i in selected_features
]


# =========================================================
# 6. FINAL RESULT
# =========================================================

print("\n========================")
print("FSNID RESULT")
print("========================")

print(
    "Selected feature indices:",
    selected_features
)

print(
    "Selected feature names:",
    selected_feature_names
)

print(
    "Number selected:",
    len(selected_features)
)