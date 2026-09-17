import pandas as pd

INPUT_FILE = "data/evaluation/golden_set_suggestions.csv"
OUTPUT_FILE = "data/evaluation/golden_set_final.csv"

df = pd.read_csv(INPUT_FILE)

intents = [
    "account_access",
    "billing",
    "delivery",
    "product_issue",
    "cancellation",
    "connectivity",
    "technical_support",
    "information_request",
    "complaint",
    "other"
]

print("=" * 70)
print("GOLDEN SET HUMAN REVIEW")
print("=" * 70)

reviewed = []

for i, row in df.iterrows():

    print()
    print("=" * 70)
    print(f"EXAMPLE {i + 1} / {len(df)}")
    print("=" * 70)

    print("\nCUSTOMER:")
    print(row["conversation_context"])

    print("\nAI SUGGESTION:")
    print(f"Intent: {row['suggested_intent']}")
    print(f"Confidence: {row['intent_confidence']:.3f}")
    print(f"Escalate: {row['suggested_escalate']}")

    print("\nINTENTS:")
    for number, intent in enumerate(intents, start=1):
        print(f"{number}. {intent}")

    while True:
        choice = input(
            "\nPress ENTER to accept suggestion, "
            "or enter 1-10 to change intent: "
        ).strip()

        if choice == "":
            final_intent = row["suggested_intent"]
            break

        if choice.isdigit() and 1 <= int(choice) <= 10:
            final_intent = intents[int(choice) - 1]
            break

        print("Please press ENTER or enter a number from 1-10.")

    while True:
        escalation = input(
            f"Escalate? [y/n] "
            f"(AI suggests {row['suggested_escalate']}): "
        ).strip().lower()

        if escalation == "":
            final_escalate = row["suggested_escalate"]
            break

        if escalation in ["y", "yes"]:
            final_escalate = "yes"
            break

        if escalation in ["n", "no"]:
            final_escalate = "no"
            break

        print("Please enter y or n.")

    reviewed.append({
        "conversation_context": row["conversation_context"],
        "intent": final_intent,
        "escalate": final_escalate
    })

    print(
        f"\n✓ Saved: {final_intent} | "
        f"escalate={final_escalate}"
    )


result = pd.DataFrame(reviewed)

result.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("=" * 70)
print("HUMAN REVIEW COMPLETE")
print("=" * 70)
print(f"Examples reviewed: {len(result)}")
print(f"Saved to: {OUTPUT_FILE}")