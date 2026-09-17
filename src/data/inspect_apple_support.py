import pandas as pd

file_path = "data/processed/apple_support.csv"

df = pd.read_csv(file_path, nrows=1000)

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 10 messages:")
print(df[["author_id", "inbound", "text"]].head(10).to_string())

print("\nInbound counts:")
print(df["inbound"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())