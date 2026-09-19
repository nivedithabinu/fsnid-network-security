import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

from src.classifier import FeatureClassifier


# =========================================================
# SETTINGS
# =========================================================

BATCH_SIZE = 512
EPOCHS = 100
LEARNING_RATE = 0.01

FSNID_FEATURES = [12, 5, 2]

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# =========================================================
# LOAD DATA
# =========================================================

X_train = np.load(
    "data/processed/NSL-KDD/X_train.npy"
)

X_test = np.load(
    "data/processed/NSL-KDD/X_test.npy"
)

y_train = np.load(
    "data/processed/NSL-KDD/y_train.npy",
    allow_pickle=True
)

y_test = np.load(
    "data/processed/NSL-KDD/y_test.npy",
    allow_pickle=True
)

print("Train:", X_train.shape)
print("Test :", X_test.shape)


# =========================================================
# LABEL ENCODING
# =========================================================

# Fit using train + test so every NSL-KDD class receives
# a valid integer ID.
encoder = LabelEncoder()

encoder.fit(
    np.concatenate([y_train, y_test])
)

y_train_encoded = encoder.transform(y_train)
y_test_encoded = encoder.transform(y_test)

num_classes = len(encoder.classes_)

print("Number of classes:", num_classes)


# =========================================================
# TRAIN + EVALUATE
# =========================================================

def run_experiment(
    X_train_data,
    X_test_data,
    seed
):

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

    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE
    )

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


# # =========================================================
# FINAL EXPERIMENT — 5 SEEDS
# =========================================================

seeds = [0, 1, 2, 3, 4]

X_train_fsnid = X_train[:, FSNID_FEATURES]
X_test_fsnid = X_test[:, FSNID_FEATURES]

all_results = []
fsnid_results = []


for seed in seeds:

    print(f"\n================================")
    print(f"SEED {seed}")
    print(f"================================")

    print("\nALL 41 FEATURES")

    acc, f1 = run_experiment(
        X_train,
        X_test,
        seed
    )

    all_results.append([acc, f1])

    print(
        f"Accuracy: {acc:.4f} | "
        f"F1: {f1:.4f}"
    )


    print("\nFSNID TOP-3 FEATURES")

    acc, f1 = run_experiment(
        X_train_fsnid,
        X_test_fsnid,
        seed
    )

    fsnid_results.append([acc, f1])

    print(
        f"Accuracy: {acc:.4f} | "
        f"F1: {f1:.4f}"
    )


# =========================================================
# MEAN + 95% CONFIDENCE INTERVAL
# =========================================================

def summarize(results):

    results = np.array(results)

    mean = results.mean(axis=0)
    std = results.std(axis=0, ddof=1)

    ci95 = 1.96 * std / np.sqrt(len(results))

    return mean, ci95


all_mean, all_ci = summarize(all_results)
fsnid_mean, fsnid_ci = summarize(fsnid_results)


print("\n\n================================")
print("FINAL RESULTS")
print("================================")

print("\nALL 41 FEATURES")

print(
    f"Accuracy: "
    f"{all_mean[0] * 100:.2f}% "
    f"± {all_ci[0] * 100:.2f}%"
)

print(
    f"Weighted F1: "
    f"{all_mean[1] * 100:.2f}% "
    f"± {all_ci[1] * 100:.2f}%"
)


print("\nFSNID TOP-3 FEATURES")

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