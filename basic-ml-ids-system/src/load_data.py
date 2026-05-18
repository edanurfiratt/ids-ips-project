import pandas as pd
from pathlib import Path

def load_dataset(file_path: Path) -> pd.DataFrame:
    print(f" Dosya okunuyor: {file_path}")

    df = pd.read_csv(
        file_path,
        sep=",",
        engine="python",
        on_bad_lines="skip"
    )

    df.columns = df.columns.astype(str).str.strip().str.replace('"', '', regex=False)

    print(f" Dosya okundu. Boyut: {df.shape}")
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    rename_map = {
        "No.": "no",
        "No": "no",
        "Time": "time",
        "Source": "source_ip",
        "Destination": "destination_ip",
        "Protocol": "protocol",
        "Length": "length",
        "Info": "info",
        "file_source": "file_source",
        "label": "label"
    }

    df = df.rename(columns=rename_map)

    required_cols = [
        "time",
        "source_ip",
        "destination_ip",
        "protocol",
        "length",
        "info",
        "label"
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Eksik kolon var: {col}")

    print(" Wireshark kolonları normalize edildi.")
    return df


def show_basic_info(df: pd.DataFrame) -> None:
    print("\n" + "="*50)
    print(" VERİ BİLGİSİ")
    print("="*50)
    print(f" Boyut: {df.shape[0]} satır, {df.shape[1]} sütun")
    print(f" Sütunlar: {df.columns.tolist()}")

    print("\nLabel dağılımı:")
    print(df["label"].value_counts())

    if "file_source" in df.columns:
        print("\n📁 Dosya kaynak dağılımı:")
        print(df["file_source"].value_counts().head(20))

    print("\n🔍 İlk 5 satır:")
    print(df.head())
    print("="*50 + "\n")