import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset_file_based(
    df: pd.DataFrame,
    test_files: list[str],
    test_size: float = 0.2,
    random_state: int = 42
):
    print("\n Dosya bazlı train/test ayrımı başlıyor...")

    if "file_source" not in df.columns:
        print(" file_source yok. Normal stratified split yapılacak.")
        return split_dataset(df, test_size, random_state)

    test_df = df[df["file_source"].isin(test_files)].copy()
    train_df = df[~df["file_source"].isin(test_files)].copy()

    if len(test_df) == 0 or len(train_df) == 0:
        print(" Test dosyaları bulunamadı. Normal split yapılacak.")
        return split_dataset(df, test_size, random_state)

    print(f" Eğitim seti: {len(train_df)} satır")
    print(f" Test seti: {len(test_df)} satır")

    print("\n Test dosyaları:")
    print(test_df["file_source"].value_counts())

    print("\n Eğitim label dağılımı:")
    print(train_df["label"].value_counts())

    print("\n Test label dağılımı:")
    print(test_df["label"].value_counts())

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def split_dataset(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    print("\n Normal split çalışıyor...")

    X = df.drop("label", axis=1)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    train_df = X_train.copy()
    train_df["label"] = y_train

    test_df = X_test.copy()
    test_df["label"] = y_test

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)