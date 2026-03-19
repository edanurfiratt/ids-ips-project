import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

KDD_PATH = os.path.join(BASE_DIR, "data", "kddcup.data")

CUSTOM_PATH = os.path.join(BASE_DIR, "data", "dataset_clean.csv")  # <- sadece bunu değiştir

KDD_COLS = ["duration", "protocol_type", "service", "src_bytes", "dst_bytes"]


def safe_read_csv(path: str) -> pd.DataFrame:
    """
    CSV okuma: ayraç/bozuk satır gibi durumlarda daha toleranslı okur.
    """
    try:
        return pd.read_csv(path)
    except Exception:
        # sep=None + engine=python ayraç tahmini yapar
        return pd.read_csv(path, sep=None, engine="python", on_bad_lines="skip")


def load_kdd_as_train(path: str) -> pd.DataFrame:
   
    full_cols = [
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
        "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
        "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
        "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
        "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
        "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
        "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
        "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
        "attack_type"
    ]

    df = pd.read_csv(path, header=None, names=full_cols)

    df["label"] = (df["attack_type"] != "normal.").astype(int)

    df = df[KDD_COLS + ["label"]].copy()
    return df


def load_custom_as_test(path: str) -> pd.DataFrame:
    
    df = safe_read_csv(path)

    required_after_rename = ["duration", "protocol_type", "service", "src_bytes", "dst_bytes", "label"]

    def try_standardize(dfx: pd.DataFrame) -> pd.DataFrame:
        dfx = dfx.copy()
        dfx.columns = [str(c).strip() for c in dfx.columns]

        dfx = dfx.rename(columns={
            "proto": "protocol_type",
            "orig_bytes": "src_bytes",
            "resp_bytes": "dst_bytes",
        })

        if "label" not in dfx.columns:
            dfx["label"] = 0

        missing = [c for c in required_after_rename if c not in dfx.columns]
        if missing:
            raise ValueError(missing)

        return dfx[required_after_rename].copy()

   
    try:
        return try_standardize(df)
    except Exception:
        pass

    df2 = pd.read_csv(path, header=None)

   
    if df2.shape[1] < 9:
        raise ValueError(f"Custom dataset beklenenden az kolonlu: {df2.shape[1]} kolon var.")

    df2 = df2.iloc[:, :9].copy()
    df2.columns = ["ts", "id.orig_h", "id.resp_h", "proto", "service", "duration", "orig_bytes", "resp_bytes", "label"]

    df2 = df2.rename(columns={
        "proto": "protocol_type",
        "orig_bytes": "src_bytes",
        "resp_bytes": "dst_bytes",
    })

    df2 = df2[required_after_rename].copy()
    return df2

def preprocess_same_way(train_df: pd.DataFrame, test_df: pd.DataFrame):

    train_df = train_df.copy()
    test_df = test_df.copy()

    
    for c in ["duration", "src_bytes", "dst_bytes"]:
        train_df[c] = pd.to_numeric(train_df[c], errors="coerce").fillna(0)
        test_df[c] = pd.to_numeric(test_df[c], errors="coerce").fillna(0)

    for c in ["protocol_type", "service"]:
        le = LabelEncoder()
        train_df[c] = le.fit_transform(train_df[c].astype(str))

        mapping = {k: i for i, k in enumerate(le.classes_)}
        test_df[c] = test_df[c].astype(str).map(mapping).fillna(-1).astype(int)

    train_df["label"] = pd.to_numeric(train_df["label"], errors="coerce").fillna(0).astype(int)
    test_df["label"] = pd.to_numeric(test_df["label"], errors="coerce").fillna(0).astype(int)

    return train_df, test_df


def main():
    print("Basladi.")

    if not os.path.exists(KDD_PATH):
        print("KDD dosyasi yok:", KDD_PATH)
        return

    if not os.path.exists(CUSTOM_PATH):
        print("Custom dosya yok:", CUSTOM_PATH)
        return

    print("KDD train yukleniyor...")
    train_df = load_kdd_as_train(KDD_PATH)
    print("Train boyut:", train_df.shape)

    print("Custom test yukleniyor...")
    test_df = load_custom_as_test(CUSTOM_PATH)
    print("Test boyut:", test_df.shape)

    train_df, test_df = preprocess_same_way(train_df, test_df)

    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"].astype(int)

    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"].astype(int)

    print("\nModel egitimi (RF + CatBoost)...")

    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    cb = CatBoostClassifier(
        iterations=300,
        learning_rate=0.1,
        depth=6,
        loss_function="Logloss",
        verbose=False,
        random_state=42
    )
    cb.fit(X_train, y_train)

    pred_rf = rf.predict(X_test)
    pred_cb = cb.predict(X_test).astype(int)

    vote = ((pred_rf + pred_cb) >= 1).astype(int)

    print("\n[RF] Test Sonuclari")
    print(confusion_matrix(y_test, pred_rf))
    print(classification_report(y_test, pred_rf))

    print("\n[CatBoost] Test Sonuclari")
    print(confusion_matrix(y_test, pred_cb))
    print(classification_report(y_test, pred_cb))

    print("\n[Voting (RF+CB)] Test Sonuclari")
    print(confusion_matrix(y_test, vote))
    print(classification_report(y_test, vote))


if __name__ == "__main__":
    main()