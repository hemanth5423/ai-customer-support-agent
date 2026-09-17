import pandas as pd

file_path = "data/raw/twcs/twcs.csv"

df = pd.read_csv(file_path, nrows=1000)

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 tweets:")
print(df["text"].head().to_string())

print("\nInbound counts:")
print(df["inbound"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())