from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression

try:
    from catboost import CatBoostClassifier

    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = (BASE_DIR / ".." / "data" / "dataset_clean.csv").resolve()

RANDOM_STATE = 42
TEST_SIZE = 0.20
CONTAMINATION = 0.10  



EXPECTED_COLS_9 = [
    "ts",
    "id.orig_h",
    "id.resp_h",
    "proto",
    "service",
    "duration",
    "orig_bytes",
    "resp_bytes",
    "label",
]



def load_dataset(file_path: Path) -> pd.DataFrame:
   
    if not file_path.exists():
        raise FileNotFoundError(f"Dosya bulunamadi: {file_path}")

    
    try:
        df = pd.read_csv(
            file_path,
            sep=",",
            engine="python",
            on_bad_lines="skip",
        )
        return df
    except Exception:
        pass

    
    df = pd.read_csv(
        file_path,
        sep=None,
        engine="python",
        on_bad_lines="skip",
    )
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    
    df.columns = [str(c).strip() for c in df.columns]

    if df.columns.tolist() == EXPECTED_COLS_9:
        return df

    
    rename_map = {
        "protocol_type": "proto",
        "src_ip": "id.orig_h",
        "dst_ip": "id.resp_h",
        "src_bytes": "orig_bytes",
        "dst_bytes": "resp_bytes",
    }
    df = df.rename(columns=rename_map)

    
    if len(df.columns) == 9 and "label" in df.columns:
        return df

    
    if len(df.columns) == 9 and ("ts" not in df.columns or "label" not in df.columns):
        
        df.columns = EXPECTED_COLS_9
        return df

    return df


def show_basic_info(df: pd.DataFrame) -> None:
    print("Ilk 5 satir:")
    print(df.head())
    print("\nSutunlar:")
    print(df.columns.tolist())
    print(f"\nBoyut: {df.shape}")


def coerce_numeric(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
   
    df = df.copy()

    if "label" not in df.columns:
        raise ValueError(
            "Label sutunu yok. Supervised egitim icin label gerekli.\n"
            "Cozum: Veri setine 'label' kolonu ekle (0=normal, 1=attack gibi) "
            "veya label ismini 'label' yap."
        )

    
    df.columns = [str(c).strip() for c in df.columns]

    
    cat_cols = [c for c in ["proto", "service", "id.orig_h", "id.resp_h"] if c in df.columns]

   
    num_cols = [c for c in ["ts", "duration", "orig_bytes", "resp_bytes"] if c in df.columns]

  
    df = coerce_numeric(df, num_cols)

    
    if df["label"].dtype == object:
        df["label"] = df["label"].astype(str).str.strip()
        df["label"] = df["label"].apply(lambda x: 0 if x in ["0", "normal", "benign"] else 1)

    df["label"] = pd.to_numeric(df["label"], errors="coerce")

    for col in cat_cols:
        enc = LabelEncoder()
        df[col] = enc.fit_transform(df[col].astype(str))

  
    df = df.dropna(subset=num_cols + ["label"]).reset_index(drop=True)

   
    df["label"] = df["label"].astype(int)

    return df


def split_dataset(df: pd.DataFrame, test_size: float, random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"] if "label" in df.columns else None,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def train_isolation_forest(train_df: pd.DataFrame, test_df: pd.DataFrame, contamination: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = [c for c in train_df.columns if c != "label"]

    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values

    print("\nIsolation Forest egitimi baslatildi...")
    print(f"contamination = {contamination}")

    iso = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    iso.fit(X_train)

    
    train_pred = iso.predict(X_train)
    test_pred = iso.predict(X_test)

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["anomaly"] = train_pred
    test_df["anomaly"] = test_pred

    print("Isolation Forest egitimi tamamlandi.")
    print("\nTrain anomaly dagilimi:")
    print(train_df["anomaly"].value_counts())
    print("\nTest anomaly dagilimi:")
    print(test_df["anomaly"].value_counts())

   
    y_true = test_df["label"].values
    y_pred = np.where(test_df["anomaly"].values == -1, 1, 0)

    print("\n[IsolationForest] TEST sonuclari:")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=2))

    return train_df, test_df


def filter_anomalies(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
  
    train_anom = train_df[train_df["anomaly"] == -1].copy()
    test_anom = test_df[test_df["anomaly"] == -1].copy()

    print("\nAnomaliler filtreleniyor...")
    print(f"Train anomali sayisi: {len(train_anom)}")
    print(f"Test anomali sayisi: {len(test_anom)}")

    return train_anom, test_anom


def train_supervised(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
   
    feature_cols = [c for c in train_df.columns if c not in ["label", "anomaly"]]

    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values

    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values

    if len(np.unique(y_train)) < 2:
        print("\n[Uyari] Train anomalilerinde tek sinif var. Supervised egitim mantikli degil.")
        return

    print("\n2. katman (supervised) basliyor...")

    if HAS_CATBOOST:
        model = CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.1,
            loss_function="Logloss",
            verbose=False,
            random_seed=RANDOM_STATE,
        )
        model.fit(X_train, y_train)
        print("Modeller egitildi. (CatBoost: var )")
        y_pred = model.predict(X_test).astype(int).ravel()
    else:
        model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        print("Modeller egitildi. (CatBoost: yok, LogisticRegression kullanildi)")
        y_pred = model.predict(X_test)

    print("\n[Kaskad Final] TEST sonuclari:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=2))



def main() -> None:
    print("Program baslatildi")
    print("Veri yukleniyor...")

    df = load_dataset(DATA_PATH)
    df = normalize_columns(df)
    show_basic_info(df)

    print("\nOn isleme basladi...")
    df = preprocess_dataset(df)
    print(f"On isleme tamamlandi. Yeni boyut: {df.shape}")

    print("\nTrain / Test ayrimi yapiliyor...")
    train_df, test_df = split_dataset(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print(f"Train boyut: {train_df.shape}  Test boyut: {test_df.shape}")

    train_df, test_df = train_isolation_forest(train_df, test_df, contamination=CONTAMINATION)

   
    train_anom, test_anom = filter_anomalies(train_df, test_df)

    
    if len(test_anom) == 0 or len(train_anom) == 0:
        print("\n[Uyari] Anomali seti bos. 2. katman calistirilamadi.")
        return

    train_supervised(train_anom, test_anom)


if __name__ == "__main__":
    main()