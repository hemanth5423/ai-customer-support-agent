import pandas as pd

file_path = "data/processed/apple_support_conversations.csv"

df = pd.read_csv(file_path)

print("Total conversation pairs:", len(df))

print("\nColumns:")
print(df.columns.tolist())

print("\nSample conversations:")

for i, row in df.head(10).iterrows():

    print("=" * 70)

    print("CUSTOMER:")
    print(row["customer_message"])

    print("\nAPPLE SUPPORT:")
    print(row["support_response"])

print("\nDataset shape:")
print(df.shape)