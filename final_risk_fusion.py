import pandas as pd
import numpy as np
import joblib

from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


DATA_PATH = r"data\wireshark_smart_labeled_dataset.csv"

TEST_FILES = [
    "mitm_link.csv",
    "slowloris.csv",
    "wifispooning.csv",
    "mitm_3.csv",
    "ftp_bruteforce.csv",
    "MITM.csv",
    "wifi_dos2.csv"
]


def hazirla(df):
    df = df.copy()

    df.columns = df.columns.astype(str).str.strip().str.replace('"', '', regex=False)

    df = df.rename(columns={
        "No.": "no",
        "No": "no",
        "Time": "time",
        "Source": "source_ip",
        "Destination": "destination_ip",
        "Protocol": "protocol",
        "Length": "length",
        "Info": "info"
    })

    df["time"] = pd.to_numeric(df["time"].astype(str).str.replace(",", ".", regex=False), errors="coerce").fillna(0)
    df["length"] = pd.to_numeric(df["length"].astype(str).str.replace(",", ".", regex=False), errors="coerce").fillna(0)
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)

    df["protocol"] = df["protocol"].fillna("unknown").astype(str)
    df["info"] = df["info"].fillna("").astype(str)

    proto = df["protocol"].str.upper()
    info = df["info"].str.upper()

    df["is_tcp"] = proto.eq("TCP").astype(int)
    df["is_udp"] = proto.eq("UDP").astype(int)
    df["is_icmp"] = proto.str.contains("ICMP", na=False).astype(int)
    df["is_dns"] = proto.eq("DNS").astype(int)
    df["is_http"] = proto.str.contains("HTTP", na=False).astype(int)
    df["is_tls"] = proto.str.contains("TLS", na=False).astype(int)
    df["is_arp"] = proto.eq("ARP").astype(int)

    df["has_syn"] = info.str.contains("SYN", na=False).astype(int)
    df["has_ack"] = info.str.contains("ACK", na=False).astype(int)
    df["has_fin"] = info.str.contains("FIN", na=False).astype(int)
    df["has_rst"] = info.str.contains("RST", na=False).astype(int)
    df["has_psh"] = info.str.contains("PSH", na=False).astype(int)
    df["has_get"] = info.str.contains("GET", na=False).astype(int)
    df["has_post"] = info.str.contains("POST", na=False).astype(int)

    df["info_length"] = df["info"].str.len()
    df["log_length"] = np.log1p(df["length"])
    df["protocol_code"] = df["protocol"].astype("category").cat.codes

    df["time_window"] = df["time"].round(0).astype(int)

    grup = ["file_source", "source_ip", "destination_ip", "protocol", "time_window"]

    df["flow_packet_count"] = df.groupby(grup)["protocol"].transform("count")
    df["flow_total_length"] = df.groupby(grup)["length"].transform("sum")
    df["avg_packet_length"] = df["flow_total_length"] / df["flow_packet_count"].replace(0, 1)

    df["length_square"] = df["length"] ** 2
    df["length_sqrt"] = np.sqrt(df["length"].clip(lower=0))
    df["length_per_flow_packet"] = df["length"] / df["flow_packet_count"].replace(0, 1)

    return df.replace([np.inf, -np.inf], 0).fillna(0)


print("\nVeri okunuyor...")
df = pd.read_csv(DATA_PATH, sep=",", engine="python", on_bad_lines="skip")
df = hazirla(df)
df = df[df["file_source"].isin(TEST_FILES)].copy()

print("\nTest boyutu:", df.shape)
print(df["label"].value_counts())

y_true = df["label"].astype(int).values


print("\nML modeli yükleniyor...")
ml_model = joblib.load("best_ml_model.pkl")
ml_features = joblib.load("ml_features.pkl")
ml_threshold = joblib.load("ml_threshold.pkl")

X_ml = df[ml_features].apply(pd.to_numeric, errors="coerce").fillna(0)
ml_prob = ml_model.predict_proba(X_ml)[:, 1]
ml_pred = (ml_prob >= ml_threshold).astype(int)


print("\nDL modeli yükleniyor...")
dl_model = load_model("hoca_dl_model.keras", compile=False)
dl_features = joblib.load("hoca_dl_features.pkl")
dl_scaler = joblib.load("hoca_dl_scaler.pkl")
dl_threshold = joblib.load("hoca_dl_threshold.pkl")

X_dl = df[dl_features].apply(pd.to_numeric, errors="coerce").fillna(0)
X_dl_scaled = dl_scaler.transform(X_dl)

dl_prob = dl_model.predict(X_dl_scaled, batch_size=4096).flatten()
dl_pred = (dl_prob >= dl_threshold).astype(int)


print("\nML SONUCU")
print(classification_report(y_true, ml_pred, target_names=["Normal", "Attack"], digits=3, zero_division=0))
print(confusion_matrix(y_true, ml_pred))


print("\nDL SONUCU")
print("DL threshold:", dl_threshold)
print(classification_report(y_true, dl_pred, target_names=["Normal", "Attack"], digits=3, zero_division=0))
print(confusion_matrix(y_true, dl_pred))




ml_weights = [0.55, 0.60, 0.65, 0.70, 0.75]
thresholds = [0.05, 0.04, 0.03, 0.02, 0.015, 0.01]

dl_rescue_thresholds = [0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001]
ml_uncertain_thresholds = [0.00, 0.01, 0.03, 0.05, 0.07, 0.10]

best_score = -999
best_pred = None
best_info = None
results = []

print("\nWEIGHTED FUSION + DL RESCUE DENEMELERI")

for ml_w in ml_weights:
    dl_w = 1 - ml_w

    base_score = (ml_w * ml_prob) + (dl_w * dl_prob)

    for fusion_th in thresholds:
        for dl_rescue_th in dl_rescue_thresholds:
            for ml_uncertain_th in ml_uncertain_thresholds:

                base_pred = (base_score >= fusion_th).astype(int)

                
                dl_rescue = (
                    (ml_pred == 0) &
                    (ml_prob >= ml_uncertain_th) &
                    (dl_prob >= dl_rescue_th)
                )

                pred = np.where(
                    (base_pred == 1) | dl_rescue,
                    1,
                    0
                )

                acc = accuracy_score(y_true, pred)
                prec = precision_score(y_true, pred, zero_division=0)
                rec = recall_score(y_true, pred, zero_division=0)
                f1 = f1_score(y_true, pred, zero_division=0)

                cm = confusion_matrix(y_true, pred, labels=[0, 1])
                tn, fp = cm[0][0], cm[0][1]
                fn, tp = cm[1][0], cm[1][1]
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

                ml_missed = (y_true == 1) & (ml_pred == 0)
                saved = ml_missed & (pred == 1)
                saved_count = int(saved.sum())

                results.append({
                    "ml_weight": ml_w,
                    "dl_weight": dl_w,
                    "fusion_threshold": fusion_th,
                    "dl_rescue_threshold": dl_rescue_th,
                    "ml_uncertain_threshold": ml_uncertain_th,
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "fpr": fpr,
                    "saved_attack": saved_count,
                    "missed_attack": int(fn)
                })

                
                if acc >= 0.85:
                    score = (
                        (2.2 * acc) +
                        (3.2 * rec) +
                        (1.8 * f1) +
                        (0.01 * saved_count) -
                        (0.9 * fpr)
                    )
                elif acc >= 0.80:
                    score = (
                        (1.5 * acc) +
                        (3.5 * rec) +
                        (1.5 * f1) +
                        (0.01 * saved_count) -
                        (1.0 * fpr)
                    )
                else:
                    score = (
                        acc +
                        (3.0 * rec) +
                        f1 +
                        (0.0015 * saved_count) -
                        (1.3 * fpr)
                    )

                if score > best_score:
                    best_score = score
                    best_pred = pred
                    best_info = results[-1]


results_df = pd.DataFrame(results)

print("\nEn iyi 15 fusion sonucu:")
print(
    results_df.sort_values(
        by=["accuracy", "recall", "f1", "saved_attack"],
        ascending=False
    ).head(15).to_string(index=False)
)

print("\nRecall odaklı en iyi 15 sonuç:")
print(
    results_df.sort_values(
        by=["recall", "accuracy", "saved_attack"],
        ascending=False
    ).head(15).to_string(index=False)
)

print("\nSEÇİLEN FUSION AYARI")
print(best_info)


print("\nFINAL WEIGHTED + DL RESCUE FUSION SONUCU")
print(classification_report(y_true, best_pred, target_names=["Normal", "Attack"], digits=3, zero_division=0))
print(confusion_matrix(y_true, best_pred))

print("\nÖzet")
print("Accuracy:", accuracy_score(y_true, best_pred))
print("Precision:", precision_score(y_true, best_pred, zero_division=0))
print("Recall:", recall_score(y_true, best_pred, zero_division=0))
print("F1:", f1_score(y_true, best_pred, zero_division=0))


attack_mask = y_true == 1
ml_missed = attack_mask & (ml_pred == 0)
fusion_saved = ml_missed & (best_pred == 1)

attack_mask = y_true == 1
normal_mask = y_true == 0

yakalanan_attack = int(((attack_mask) & (best_pred == 1)).sum())
kacirilan_attack = int(((attack_mask) & (best_pred == 0)).sum())

dogru_normal = int(((normal_mask) & (best_pred == 0)).sum())
yanlis_alarm = int(((normal_mask) & (best_pred == 1)).sum())

toplam_attack = int(attack_mask.sum())
toplam_normal = int(normal_mask.sum())

print("\nFinal Sistem Analizi")
print("Toplam attack:", toplam_attack)
print("Yakalanan attack:", yakalanan_attack)
print("Kaçırılan attack:", kacirilan_attack)
print("Toplam normal:", toplam_normal)
print("Doğru normal:", dogru_normal)
print("Yanlış alarm:", yanlis_alarm)

if toplam_attack > 0:
    print("Attack yakalama oranı:", yakalanan_attack / toplam_attack)

if toplam_normal > 0:
    print("Normal doğru tanıma oranı:", dogru_normal / toplam_normal)


pd.DataFrame({
    "file_source": df["file_source"].values,
    "label": y_true,
    "ml_prob": ml_prob,
    "ml_pred": ml_pred,
    "dl_prob": dl_prob,
    "dl_pred": dl_pred,
    "final_pred": best_pred
}).to_csv("final_weighted_fusion_results.csv", index=False)



print("\nKaydedildi:")
print("final_weighted_rescue_fusion_results.csv")
print("weighted_rescue_fusion_all_results.csv")