import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


INPUT_FILE = "data/evaluation/golden_set_final.csv"


def main():

    print("=" * 70)
    print("MAJORITY CLASS BASELINE")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE)

    df = df.dropna(subset=["conversation_context", "intent"])

    split = int(len(df) * 0.8)

    train = df.iloc[:split]
    test = df.iloc[split:]

    # Find the most common intent in training data
    majority_intent = train["intent"].value_counts().idxmax()

    print(f"\nTraining examples: {len(train)}")
    print(f"Testing examples: {len(test)}")

    print(f"\nMost common intent: {majority_intent}")

    # Predict the same intent for every test example
    predictions = [majority_intent] * len(test)

    accuracy = accuracy_score(
        test["intent"],
        predictions
    )

    print("\n" + "=" * 70)
    print("MAJORITY BASELINE RESULTS")
    print("=" * 70)

    print(f"\nIntent Accuracy: {accuracy:.3f}")

    print("\nClassification Report:")

    print(
        classification_report(
            test["intent"],
            predictions,
            zero_division=0
        )
    )


if __name__ == "__main__":
    main()