import pandas as pd
import numpy as np
COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label"
]

def load_kdd_dataset(file_path: str) -> pd.DataFrame:
  
    df = pd.read_csv(
        file_path,
        names=COLUMN_NAMES
    )
    
    return df

def show_basic_info(df: pd.DataFrame) -> None:
    print("\n İlk 5 kayıt:")
    print(df.head())

    print("\n Veri boyutu:")
    print(df.shape)

if __name__ == "__main__":
    DATA_PATH = "../data/kddcup.data"

    kdd_df = load_kdd_dataset(DATA_PATH)
    show_basic_info(kdd_df)