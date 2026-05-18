import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

   
    for col in ["duration", "orig_bytes", "resp_bytes", "ts"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

 
    cat_cols = []
    for c in ["proto", "service", "id.orig_h", "id.resp_h"]:
        if c in df.columns:
            cat_cols.append(c)

    for col in cat_cols:
        df[col] = df[col].astype(str).fillna("NA")
        df[col] = LabelEncoder().fit_transform(df[col])

    if "label" in df.columns:
        df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
        df["label"] = df["label"].apply(lambda x: 1 if x != 0 else 0)
    else:
        raise ValueError("label kolonu yok. Veri setinde label olmalı.")

   
    df = df.dropna()

    return df