import pandas as pd

df = pd.read_csv("data/final_dataset_cleanN.csv", sep=None, engine="python")

print("Kolonlar:")
print(df.columns.tolist())

print("\nLabel dağılımı:")
print(df["label"].value_counts())

if "source_file" in df.columns:
    print("\nSource file dağılımı:")
    print(df["source_file"].value_counts())
else:
    print("\nsource_file kolonu YOK.")