import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def train_isolation_forest(train_df: pd.DataFrame, contamination: float = 0.10):
    print("\n" + "="*50)
    print("🌲 ISOLATION FOREST EĞİTİMİ")
    print("="*50)

    normal_train = train_df[train_df["label"] == 0].copy()

    print(f" Toplam eğitim verisi: {len(train_df)}")
    print(f" Normal trafik sayısı: {len(normal_train)}")

    if len(normal_train) < 10:
        print("⚠️ Normal trafik çok az. Tüm eğitim verisi kullanılacak.")
        normal_train = train_df.copy()

    feature_cols = [
        c for c in normal_train.columns
        if c not in ["label", "anomaly", "file_source"]
    ]

    X_normal = normal_train[feature_cols]
    X_normal = X_normal.apply(pd.to_numeric, errors="coerce").fillna(0)
    X_normal = X_normal.replace([np.inf, -np.inf], 0)

    print(f" Özellik sayısı: {X_normal.shape[1]}")
    print(f" Contamination: {contamination}")

    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_normal)

    print(" Isolation Forest eğitimi tamamlandı.")
    return model


def predict_anomalies(model, df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        c for c in df.columns
        if c not in ["label", "anomaly", "file_source"]
    ]

    X = df[feature_cols]
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    X = X.replace([np.inf, -np.inf], 0)

    predictions = model.predict(X)

    result_df = df.copy()
    result_df["anomaly"] = predictions

    anomaly_count = (predictions == -1).sum()
    normal_count = (predictions == 1).sum()

    print(f"    Anomali: {anomaly_count} ({anomaly_count / len(df) * 100:.1f}%)")
    print(f"    Normal: {normal_count} ({normal_count / len(df) * 100:.1f}%)")

    return result_df