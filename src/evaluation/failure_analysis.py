from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data/evaluation/evaluation_results_corrected.csv"
OUTPUT = ROOT / "data/evaluation/failure_cases_corrected.csv"


def main():
    df = pd.read_csv(INPUT)
    intent_wrong = df.expected_intent != df.predicted_intent
    escalation_wrong = df.expected_escalate != df.predicted_escalate
    failures = df[intent_wrong | escalation_wrong].copy()

    def kind(row):
        i = row.expected_intent != row.predicted_intent
        e = row.expected_escalate != row.predicted_escalate
        return "intent_and_escalation" if i and e else "intent_error" if i else "escalation_error"

    failures["failure_type"] = failures.apply(kind, axis=1)
    failures.to_csv(OUTPUT, index=False)

    print(f"Examples: {len(df)}")
    print(f"Failures: {len(failures)}")
    print(failures.failure_type.value_counts())
    print("\nMost common intent confusions:")
    print(failures[failures.failure_type.str.contains("intent")][
        ["expected_intent", "predicted_intent"]
    ].value_counts().head(10))
    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
