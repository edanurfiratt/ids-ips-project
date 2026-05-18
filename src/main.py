from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from config import *
from load_data import load_dataset, normalize_columns, show_basic_info
from preprocessing import preprocess_dataset
from feature_engineering import add_features
from split_data import split_dataset_file_based
from isolation_forest_model import train_isolation_forest, predict_anomalies
from supervised_model import train_supervised_model


def main():
    print("\n" + "="*60)
    print(" IDS/IPS WIRESHARK TABANLI ANOMALİ TESPİT SİSTEMİ")
    print("="*60)

    print("\n ADIM 1: Veri yükleniyor...")

    if not DATA_PATH.exists():
        print(f"\n HATA: Dosya bulunamadı: {DATA_PATH}")
        return

    df = load_dataset(DATA_PATH)
    df = normalize_columns(df)
    show_basic_info(df)

    print("\n🔧 ADIM 2: Veri ön işleme...")
    df = preprocess_dataset(df)

    print("\n ADIM 3: Ek özellik mühendisliği...")
    df = add_features(df)

    print("\n ADIM 4: Dosya bazlı train/test ayrımı...")
    train_df, test_df = split_dataset_file_based(
        df,
        test_files=TEST_FILES,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    print("\n ADIM 5: Isolation Forest eğitiliyor...")
    if_model = train_isolation_forest(
        train_df,
        contamination=CONTAMINATION
    )

    print("\n ADIM 6: Isolation Forest tahmini yapılıyor...")
    train_df = predict_anomalies(if_model, train_df)
    test_df = predict_anomalies(if_model, test_df)

    print("\n ADIM 7: Denetimli modeller eğitiliyor...")
    train_supervised_model(train_df, test_df)

    print("\n" + "="*60)
    print(" İŞLEM TAMAMLANDI")
    print("="*60)


if __name__ == "__main__":
    main()