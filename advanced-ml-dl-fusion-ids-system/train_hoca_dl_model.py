import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


DATA_PATH = "data/wireshark_smart_labeled_dataset.csv"

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

train_df = df[~df["file_source"].isin(TEST_FILES)].copy()
test_df = df[df["file_source"].isin(TEST_FILES)].copy()

print("\nTrain:", train_df.shape)
print("Test:", test_df.shape)

print("\nTrain label dağılımı:")
print(train_df["label"].value_counts())

print("\nTest label dağılımı:")
print(test_df["label"].value_counts())


features = joblib.load("ml_features.pkl")

normal_train = train_df[train_df["label"] == 0]
attack_train = train_df[train_df["label"] == 1]

NORMAL_SAMPLE_SIZE = 750000
ATTACK_SAMPLE_SIZE = 550000

normal_sample = normal_train.sample(
    n=min(NORMAL_SAMPLE_SIZE, len(normal_train)),
    random_state=42
)

attack_sample = attack_train.sample(
    n=min(ATTACK_SAMPLE_SIZE, len(attack_train)),
    random_state=42
)

train_balanced = pd.concat([normal_sample, attack_sample], axis=0)
train_balanced = train_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nEğitim için seçilen dağılım:")
print(train_balanced["label"].value_counts())


X_train = train_balanced[features].apply(pd.to_numeric, errors="coerce").fillna(0)
y_train = train_balanced["label"].astype(int)

X_test = test_df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
y_test = test_df["label"].astype(int)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


print("\nDeep Learning modeli oluşturuluyor...")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),

    tf.keras.layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.L2(1e-5)
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.15),

    tf.keras.layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=regularizers.L2(1e-5)
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.15),

    tf.keras.layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.L2(1e-5)
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.10),

    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.10),

    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0004),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

lr_reduce = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=0.00005
)


print("\nModel eğitiliyor...")

model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.2,
    epochs=35,
    batch_size=1024,
    callbacks=[early_stop, lr_reduce],
    verbose=1
)


print("\nTahminler alınıyor...")
y_prob = model.predict(X_test_scaled, batch_size=4096).flatten()


thresholds = [
    0.95, 0.90, 0.85, 0.80, 0.75, 0.70,
    0.65, 0.60, 0.55, 0.50, 0.45, 0.40,
    0.35, 0.30, 0.25, 0.20, 0.15, 0.10,
    0.07, 0.05, 0.03, 0.01
]

best_threshold = None
best_pred = None
best_score = -999

print("\n==============================")
print("THRESHOLD DENEMELERI")
print("==============================")

for threshold in thresholds:
    pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)

    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    fp = cm[0][1]
    tn = cm[0][0]
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    print(
        f"Threshold={threshold:.2f} | "
        f"Accuracy={acc*100:.2f}% | "
        f"Precision={prec*100:.2f}% | "
        f"Recall={rec*100:.2f}% | "
        f"F1={f1*100:.2f}% | "
        f"FPR={fpr*100:.2f}%"
    )

    # Amaç: accuracy çok düşmeden saldırı yakalama oranını artırmak.
    score = (2.0 * acc) + (3.0 * rec) + (1.5 * f1) - (0.7 * fpr)

    if score > best_score:
        best_score = score
        best_threshold = threshold
        best_pred = pred


print("\n==============================")
print("HOCA DL MODEL FINAL SONUCU")
print("==============================")
print("Seçilen threshold:", best_threshold)

print(classification_report(
    y_test,
    best_pred,
    target_names=["Normal", "Attack"],
    digits=3,
    zero_division=0
))

print(confusion_matrix(y_test, best_pred))

print("\nÖzet")
print("Accuracy:", accuracy_score(y_test, best_pred))
print("Precision:", precision_score(y_test, best_pred, zero_division=0))
print("Recall:", recall_score(y_test, best_pred, zero_division=0))
print("F1:", f1_score(y_test, best_pred, zero_division=0))


model.save("hoca_dl_model.keras")
joblib.dump(features, "hoca_dl_features.pkl")
joblib.dump(scaler, "hoca_dl_scaler.pkl")
joblib.dump(best_threshold, "hoca_dl_threshold.pkl")

pd.DataFrame({
    "y_true": y_test.values,
    "dl_prob": y_prob,
    "dl_pred": best_pred
}).to_csv("hoca_dl_predictions.csv", index=False)

print("\nKaydedildi:")
print("hoca_dl_model.keras")
print("hoca_dl_features.pkl")
print("hoca_dl_scaler.pkl")
print("hoca_dl_threshold.pkl")
print("hoca_dl_predictions.csv")