import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    

   
    df["label"] = df["label"].apply(
        lambda x: 0 if x == "normal." else 1
    )

   
    categorical_cols = ["protocol_type", "service", "flag"]

    encoder = LabelEncoder()
    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col])

   
    numeric_cols = df.columns.drop("label")

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    
    return df
