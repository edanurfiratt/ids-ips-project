import pandas as pd


def filter_anomalies(df: pd.DataFrame) -> pd.DataFrame:
  
    print("Anomaliler filtreleniyor...")

    anomalous_df = df[df["anomaly"] == -1]

    print("Anomali filtreleme tamamlandı.")
    print(f"Anomali veri sayısı: {anomalous_df.shape[0]}")

    return anomalous_df
