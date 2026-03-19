import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(df: pd.DataFrame, test_size: float = 0.2):
    """
    Veri setini eğitim ve test olarak böler.
    """

    print("Veri bölme işlemi başlatıldı...")

   
    X = df.drop("label", axis=1)
    y = df["label"]

    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    print("Veri bölme işlemi tamamlandı.")
    print(f"Eğitim seti boyutu: {X_train.shape}")
    print(f"Test seti boyutu: {X_test.shape}")

    return X_train, X_test, y_train, y_test
