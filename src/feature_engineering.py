import pandas as pd
import numpy as np

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    print(" Ek özellik mühendisliği yapılıyor...")

    df["length_square"] = df["length"] ** 2
    df["length_sqrt"] = np.sqrt(df["length"].clip(lower=0))
    df["length_per_flow_packet"] = df["length"] / df["flow_packet_count"].replace(0, 1)

    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)

    print(f" Ek özellikler tamamlandı. Toplam sütun: {df.shape[1]}")
    return df