import numpy as np 
import pandas as pd 

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

TRAINING_PATH = BASE_DIR / "data" / "raw" / "KDDTrain+.txt"
TESTING_PATH = BASE_DIR / "data" / "raw" / "KDDTest+.txt"
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "NSL-KDD"

COLUMNS = [
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
    "dst_host_srv_rerror_rate",
    "label",
    "difficulty"
]

def load():
    train_df = pd.read_csv(TRAINING_PATH, names=COLUMNS)
    test_df = pd.read_csv(TESTING_PATH, names=COLUMNS)
    return train_df, test_df
    
def separate_features(train_df, test_df):
    X_train = train_df.drop(columns=["label", "difficulty"])
    y_train = train_df["label"]
    
    X_test = test_df.drop(columns=["label", "difficulty"])
    y_test = test_df["label"]
    
    return X_train, y_train, X_test, y_test

CATEGORICAL_FEATURES = ["protocol_type", "service", "flag"]

def encoding(X_train, X_test):
    mappings = {}
    for i in CATEGORICAL_FEATURES:
        categories = sorted(X_train[i].unique())
        mappings[i] = {
            j: k for k, j in enumerate(categories)
        }

        X_train[i] = X_train[i].map(mappings[i])
        X_test[i] = X_test[i].map(mappings[i])

    return X_train, X_test, mappings

def check_data_quality(train_df, test_df, X_train, X_test):
    print("\n" + "=" * 20)
    print("DATA QUALITY CHECKS")
    print("=" * 20)

    train_missing = train_df.isna().sum().sum()
    test_missing = test_df.isna().sum().sum()

    print("Missing values: ")
    print("Train:", train_missing)
    print("Test :", test_missing)

    train_duplicates = train_df.duplicated().sum()
    test_duplicates = test_df.duplicated().sum()

    print("Duplicate rows:")
    print("Train:", train_duplicates)
    print("Test :", test_duplicates)

    train_inf = np.isinf(X_train).sum().sum()
    test_inf = np.isinf(X_test).sum().sum()

    print("Infinite values:")
    print("Train:", train_inf)
    print("Test :", test_inf)

    print("Blank categorical values:")

    for i in CATEGORICAL_FEATURES:
        train_blank = (train_df[i].astype(str).str.strip().eq("").sum())
        test_blank = (test_df[i].astype(str).str.strip().eq("").sum())
        print(f"{i} - " f"Train: {train_blank}, " f"Test: {test_blank}")
        
def normalization(X_train, X_test):
    min_train = X_train.min()
    max_train = X_train.max()
    
    range_train = max_train-min_train
    range_train = range_train.replace(0, 1)

    X_train_scaled = (X_train-min_train)/range_train
    X_test_scaled = (X_test-min_train)/range_train

    X_train_scaled = X_train_scaled.fillna(0)
    X_test_scaled = X_test_scaled.fillna(0)

    return X_train_scaled, X_test_scaled

def final(X_train, X_test, y_train, y_test):
    print("\n" + "=" * 20)
    print("FINAL VALIDATION")
    print("=" * 20)

    print("Shapes:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_test :", X_test.shape)
    print("y_test :", y_test.shape)

    print("Data types:")
    print("X_train numeric: ", X_train.select_dtypes(exclude=["number"]).columns.tolist() == [])
    print("X_test numeric: ", X_test.select_dtypes(exclude=["number"]).columns.tolist() == [])

    print("Missing values:")
    print("X_train:", X_train.isna().sum().sum())
    print("X_test :", X_test.isna().sum().sum())

    print("Infinite values:")
    print("X_train:", np.isinf(X_train).sum().sum())
    print("X_test :", np.isinf(X_test).sum().sum())

    print("Feature ranges:")
    print("X_train min:", X_train.min().min())
    print("X_train max:", X_train.max().max())

    print("X_test min:", X_test.min().min())
    print("X_test max:", X_test.max().max())

    print("Number of training classes:", y_train.nunique())
    print("Number of testing classes:", y_test.nunique())

    print("Test labels unseen during training:")
    unseen_labels = sorted(set(y_test.unique()) - set(y_train.unique()))

    print(unseen_labels)
    print("Number of unseen test classes:", len(unseen_labels))

def save_data(X_train, X_test, y_train, y_test):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    np.save(PROCESSED_DIR / "X_train.npy", X_train.to_numpy())
    np.save(PROCESSED_DIR / "X_test.npy", X_test.to_numpy())
    np.save(PROCESSED_DIR / "y_train.npy", y_train.to_numpy())
    np.save(PROCESSED_DIR / "y_test.npy", y_test.to_numpy())

    print("\n" + "=" * 20)
    print("PROCESSED DATA SAVED")
    print("=" * 20)

    print("Saved to:", PROCESSED_DIR)


if __name__ == "__main__":
    train_df, test_df = load()

    print("Training shape: ", train_df.shape)
    print("Testing shape : ", test_df.shape)

    X_train, y_train, X_test, y_test = separate_features(train_df, test_df)

    print("\nX_train shape: ", X_train.shape)
    print("y_train shape: ", y_train.shape)
    print("X_test shape : ", X_test.shape)
    print("y_test shape : ", y_test.shape)

    X_train, X_test, mappings = encoding(X_train, X_test)

    print("Categorical mappings: ")
    for i, j in mappings.items():
        print(i,":", len(j), "categories")

    print("Non-numeric columns in X_train:", X_train.select_dtypes(exclude=["number"]).columns.tolist())
    print("Non-numeric columns in X_test:", X_test.select_dtypes(exclude=["number"]).columns.tolist())

    check_data_quality(train_df, test_df, X_train, X_test)

    X_train_processed, X_test_processed = normalization(X_train, X_test)
    final(X_train_processed, X_test_processed, y_train, y_test)
    save_data(X_train_processed, X_test_processed, y_train, y_test)

if __name__ == "__main__":
    train_df, test_df = load()

    X_train, y_train, X_test, y_test = separate_features(train_df, test_df)
    X_train, X_test, mappings = encoding(X_train, X_test)

    print("Training shape:", train_df.shape)
    print("Testing shape:", test_df.shape)

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    print("Categorical mappings: ")
    for i, j in mappings.items():
        print(i, ":", len(j), "categories")

    print("Non-numeric columns in X_train:", X_train.select_dtypes(exclude=["number"]).columns.tolist())
    print("Non-numeric columns in X_test:", X_test.select_dtypes(exclude=["number"]).columns.tolist())