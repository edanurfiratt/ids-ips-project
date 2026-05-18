import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    print(" Wireshark veri ön işleme başlıyor...")
    print(f"   Başlangıç satır sayısı: {len(df)}")

    df.columns = df.columns.astype(str).str.strip()

    needed = ["time", "source_ip", "destination_ip", "protocol", "length", "info", "label"]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Eksik kolon: {col}")

    keep_cols = needed.copy()
    if "file_source" in df.columns:
        keep_cols.append("file_source")

    df = df[keep_cols].copy()

    df["time"] = pd.to_numeric(
        df["time"].astype(str).str.replace(",", ".", regex=False).str.replace("########", "0", regex=False),
        errors="coerce"
    ).fillna(0)

    df["length"] = pd.to_numeric(
        df["length"].astype(str).str.replace(",", ".", regex=False).str.replace("########", "0", regex=False),
        errors="coerce"
    ).fillna(0)

    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(1).astype(int)
    df["label"] = df["label"].apply(lambda x: 0 if x == 0 else 1)

    df["protocol"] = df["protocol"].fillna("unknown").astype(str).str.strip()
    df["info"] = df["info"].fillna("").astype(str)

    df = df[df["length"] >= 0]

    proto_upper = df["protocol"].str.upper()
    info_upper = df["info"].str.upper()

    df["is_tcp"] = proto_upper.eq("TCP").astype(int)
    df["is_udp"] = proto_upper.eq("UDP").astype(int)
    df["is_icmp"] = proto_upper.str.contains("ICMP", na=False).astype(int)
    df["is_dns"] = proto_upper.eq("DNS").astype(int)
    df["is_http"] = proto_upper.str.contains("HTTP", na=False).astype(int)
    df["is_tls"] = proto_upper.str.contains("TLS", na=False).astype(int)
    df["is_arp"] = proto_upper.eq("ARP").astype(int)

    df["has_syn"] = info_upper.str.contains("SYN", na=False).astype(int)
    df["has_ack"] = info_upper.str.contains("ACK", na=False).astype(int)
    df["has_fin"] = info_upper.str.contains("FIN", na=False).astype(int)
    df["has_rst"] = info_upper.str.contains("RST", na=False).astype(int)
    df["has_psh"] = info_upper.str.contains("PSH", na=False).astype(int)
    df["has_get"] = info_upper.str.contains("GET", na=False).astype(int)
    df["has_post"] = info_upper.str.contains("POST", na=False).astype(int)

    df["info_length"] = df["info"].str.len()
    df["log_length"] = np.log1p(df["length"])

    df["time_window"] = df["time"].astype(float).round(0).astype(int)

    group_cols = ["source_ip", "destination_ip", "protocol", "time_window"]
    df["flow_packet_count"] = df.groupby(group_cols)["protocol"].transform("count")
    df["flow_total_length"] = df.groupby(group_cols)["length"].transform("sum")
    df["avg_packet_length"] = df["flow_total_length"] / df["flow_packet_count"].replace(0, 1)

    le = LabelEncoder()
    df["protocol_code"] = le.fit_transform(df["protocol"])

    final_cols = [
        "time",
        "length",
        "log_length",
        "protocol_code",
        "is_tcp",
        "is_udp",
        "is_icmp",
        "is_dns",
        "is_http",
        "is_tls",
        "is_arp",
        "has_syn",
        "has_ack",
        "has_fin",
        "has_rst",
        "has_psh",
        "has_get",
        "has_post",
        "info_length",
        "flow_packet_count",
        "flow_total_length",
        "avg_packet_length",
        "label"
    ]

    if "file_source" in df.columns:
        final_cols.append("file_source")

    df = df[final_cols].copy()
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)

    print(f" Ön işleme tamamlandı. Son boyut: {df.shape}")
    print(f"    Normal={(df['label'] == 0).sum()} / Attack={(df['label'] == 1).sum()}")

    return df