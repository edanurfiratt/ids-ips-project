import pandas as pd

def load_dataset(file_path: str) -> pd.DataFrame:
   
    df = pd.read_csv(
        file_path,
        header=None,
        sep=",",
        engine="python",
        on_bad_lines="skip",
    )

    col_count = df.shape[1]

  
    if col_count == 8:
        df.columns = [
            "ts",
            "id.orig_h",
            "id.resp_h",
            "proto",
            "service",
            "duration",
            "orig_bytes",
            "label",
        ]
        return df

   
    if col_count == 9:
        df.columns = [
            "ts",
            "id.orig_h",
            "id.resp_h",
            "proto",
            "service",
            "duration",
            "orig_bytes",
            "resp_bytes",
            "label",
        ]
        return df

    raise ValueError(f"Beklenmeyen sütun sayısı: {col_count}. Dosyada kaç kolon var kontrol et.")

def show_basic_info(df: pd.DataFrame) -> None:
    print("Ilk 5 satir:")
    print(df.head())
    print("\nSutunlar:")
    print(df.columns.tolist())
    print("\nBoyut:", df.shape)