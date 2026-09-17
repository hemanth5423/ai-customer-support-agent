import pandas as pd

FILE_PATH = "data/evaluation/golden_set.csv"

df = pd.read_csv(FILE_PATH)

intents = {
    "1": "account_access",
    "2": "billing_payment",
    "3": "purchase_order",
    "4": "refund_return",
    "5": "product_issue",
    "6": "connectivity",
    "7": "technical_support",
    "8": "information_request",
    "9": "complaint",
    "10": "other"
}

print("=" * 70)
print("APPLE SUPPORT GOLDEN SET LABELING")
print("=" * 70)

for index, row in df.iterrows():

    print("\n" + "=" * 70)
    print(f"EXAMPLE {index + 1} / {len(df)}")
    print("=" * 70)

    print("\nCONVERSATION HISTORY:")
    print(row["conversation_context"])

    print("\nCUSTOMER:")
    print(row["customer_message"])

    print("\nHISTORICAL APPLE RESPONSE:")
    print(row["support_response"])

    print("\nINTENTS:")
    for number, intent in intents.items():
        print(f"{number}. {intent}")

    while True:
        choice = input(
            "\nChoose intent (1-10), or s to skip: "
        ).strip().lower()

        if choice == "s":
            df.at[index, "intent"] = "skipped"
            df.at[index, "escalate"] = "skipped"
            break

        if choice in intents:
            df.at[index, "intent"] = intents[choice]
            break

        print("Invalid choice. Enter 1-10 or s.")

    if choice == "s":
        continue

    while True:
        escalation = input(
            "Escalate to human? (y/n): "
        ).strip().lower()

        if escalation in ["y", "n"]:
            df.at[index, "escalate"] = escalation
            break

        print("Please enter y or n.")

    # Save after every example
    df.to_csv(FILE_PATH, index=False)

print("\n" + "=" * 70)
print("LABELING COMPLETE")
print("=" * 70)
print(f"Saved to: {FILE_PATH}")