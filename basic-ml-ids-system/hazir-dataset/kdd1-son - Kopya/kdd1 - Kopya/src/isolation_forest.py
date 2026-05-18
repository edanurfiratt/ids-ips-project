import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def fit_isolation_forest(train_df, contamination=0.10):
    train_df = train_df[train_df["label"].astype(int) == 0].copy()
    X_train = train_df.drop(columns=["label"], errors="ignore")

    model = IsolationForest(
        n_estimators=120,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train)
    print("contamination =", contamination)
    print("Isolation Forest eğitimi tamamlandı.")
    return model


def add_anomaly_column(model, df):
    X = df.drop(columns=["label"], errors="ignore")
    preds = model.predict(X)   # 1 normal, -1 anomali
    out = df.copy()
    out["anomaly"] = preds
    return out


def run_isolation_forest(train_df, test_df, contamination=0.10):
    model = fit_isolation_forest(train_df, contamination=contamination)

    train_out = add_anomaly_column(model, train_df)
    test_out = add_anomaly_column(model, test_df)

    print("\nTrain anomaly dağılımı:")
    print(train_out["anomaly"].value_counts())

    print("\nTest anomaly dağılımı:")
    print(test_out["anomaly"].value_counts())

    return model, train_out, test_out

