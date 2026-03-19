import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from load_data import load_kdd_dataset
from preprocessing import preprocess_dataset
from filter_anomalies import filter_anomalies
from isolation_forest import run_isolation_forest
from supervised_models import run_supervised_extended, predict_vote_extended


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "kddcup.data")


def main():
    print("Program başlatıldı")

    if not os.path.exists(DATA_PATH):
        print("Hata: Veri dosyası bulunamadı:", DATA_PATH)
        return

    print("Veri seti yükleniyor")
    df = load_kdd_dataset(DATA_PATH)
    print("Veri seti yüklendi")

    print("\nVeri boyutu:")
    print(df.shape)

    print("Ön işleme başlandı...")
    df = preprocess_dataset(df)
    print("Ön işleme tamamlandı.")

    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["label"].astype(int)
    )

    print("Isolation Forest eğitimi başlatıldı...")
    model, train_df, test_df = run_isolation_forest(
        train_df,
        test_df,
        contamination=0.10
    )

    y_true_test = test_df["label"].astype(int)
    y_pred_if_test = (test_df["anomaly"] == -1).astype(int)

    print("\n[IsolationForest] TEST sonuçları:")
    print(confusion_matrix(y_true_test, y_pred_if_test))
    print(classification_report(y_true_test, y_pred_if_test))

    print("\nAnomaliler filtreleniyor...")
    train_anom = filter_anomalies(train_df)
    test_anom = filter_anomalies(test_df)

    print("Train anomali sayısı:", len(train_anom))
    print("Test anomali sayısı:", len(test_anom))

    print("\n2. katman (supervised) başlıyor...")
    rf, et, gb, cb = run_supervised_extended(train_anom)

    final_pred = np.zeros(len(test_df), dtype=int)
    mask = (test_df["anomaly"] == -1)
    final_pred[mask] = predict_vote_extended(test_df[mask], rf, et, gb, cb)

    print("\n[Kaskad Final] TEST sonuçları:")
    print(confusion_matrix(test_df["label"].astype(int), final_pred))
    print(classification_report(test_df["label"].astype(int), final_pred))


if __name__ == "__main__":
    main()