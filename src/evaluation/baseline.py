import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


INPUT_FILE = "data/evaluation/golden_set_final.csv"


def main():

    print("=" * 70)
    print("TF-IDF + LOGISTIC REGRESSION BASELINE")
    print("=" * 70)

    # Load evaluation dataset
    df = pd.read_csv(INPUT_FILE)

    # Remove missing values
    df = df.dropna(subset=["conversation_context", "intent"])

    X = df["conversation_context"]
    y = df["intent"]

    # Split data
    split = int(len(df) * 0.8)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    print(f"\nTraining examples: {len(X_train)}")
    print(f"Testing examples: {len(X_test)}")

    # TF-IDF converts text into numerical features
    # Logistic Regression learns to classify those features
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                max_features=10000
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000
            )
        )
    ])

    print("\nTraining baseline...")

    model.fit(X_train, y_train)

    print("Training complete.")

    # Predict intents
    predictions = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, predictions)

    print("\n" + "=" * 70)
    print("BASELINE RESULTS")
    print("=" * 70)

    print(f"\nIntent Accuracy: {accuracy:.3f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )


if __name__ == "__main__":
    main()