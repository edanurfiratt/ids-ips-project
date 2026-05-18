from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = (BASE_DIR / ".." / "data" / "wireshark_smart_labeled_dataset.csv").resolve()

RANDOM_STATE = 42
TEST_SIZE = 0.20

CONTAMINATION = 0.20
IF_N_ESTIMATORS = 100

TEST_FILES = [
    "mitm_3.csv",
    "mitm_link.csv",
    "MITM.csv",
    "ftp_bruteforce.csv",
    "slowloris.csv",
    "wifispooning.csv",
    "wifi_dos2.csv"
]