"""Two intent baselines on the same labelled set.

Uses an 80/20 stratified split only for baseline training; the final golden
set remains the held-out benchmark for the agent when the full workflow is run.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[2]


def main():
    df = pd.read_csv(ROOT / "data/evaluation/golden_set_final.csv")
    df = df.dropna(subset=["conversation_context", "intent"])
    df = df[df.intent.ne("skipped")]

    x_train, x_test, y_train, y_test = train_test_split(
        df.conversation_context,
        df.intent,
        test_size=0.20,
        random_state=42,
        stratify=df.intent,
    )

    majority = y_train.value_counts().idxmax()
    majority_acc = accuracy_score(y_test, [majority] * len(y_test))

    tfidf = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000)),
        ("lr", LogisticRegression(max_iter=1000, C=2)),
    ])
    tfidf.fit(x_train, y_train)
    pred = tfidf.predict(x_test)
    tfidf_acc = accuracy_score(y_test, pred)

    print(f"Majority baseline: {majority_acc:.3f} ({majority})")
    print(f"TF-IDF + logistic regression: {tfidf_acc:.3f}")
    print("Split: 80/20 stratified, random_state=42")


if __name__ == "__main__":
    main()
