import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)


# ============================================================
# FILE PATH
# ============================================================

INPUT_FILE = "data/evaluation/evaluation_results.csv"


# ============================================================
# NORMALIZE INTENT VALUES
# ============================================================

def normalize_intent(value):
    """
    Convert intent values into a consistent string format.
    Missing values become 'other'.
    """

    if pd.isna(value):
        return "other"

    value = str(value).strip().lower()

    if value == "":
        return "other"

    return value


# ============================================================
# NORMALIZE ESCALATION VALUES
# ============================================================

def normalize_escalation(value):
    """
    Convert all escalation representations into:

        yes = escalate
        no  = do not escalate

    Supports:
        y / yes / true / 1
        n / no / false / 0
    """

    if pd.isna(value):
        return "no"

    value = str(value).strip().lower()

    if value in ["y", "yes", "true", "1"]:
        return "yes"

    if value in ["n", "no", "false", "0"]:
        return "no"

    return "no"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD RESULTS
    # --------------------------------------------------------

    df = pd.read_csv(INPUT_FILE)

    print(f"\nEvaluation rows: {len(df)}")

    print("\nColumns found:")
    print(list(df.columns))

    # --------------------------------------------------------
    # CHECK REQUIRED COLUMNS
    # --------------------------------------------------------

    required_columns = [
        "expected_intent",
        "predicted_intent",
        "expected_escalate",
        "predicted_escalate"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        print("\nERROR: Missing required columns:")
        print(missing_columns)

        print("\nAvailable columns:")
        print(list(df.columns))

        return

    # --------------------------------------------------------
    # NORMALIZE DATA
    # --------------------------------------------------------

    df["expected_intent"] = (
        df["expected_intent"]
        .apply(normalize_intent)
    )

    df["predicted_intent"] = (
        df["predicted_intent"]
        .apply(normalize_intent)
    )

    df["expected_escalate"] = (
        df["expected_escalate"]
        .apply(normalize_escalation)
    )

    df["predicted_escalate"] = (
        df["predicted_escalate"]
        .apply(normalize_escalation)
    )

    # ========================================================
    # INTENT CLASSIFICATION
    # ========================================================

    print("\n\n" + "=" * 70)
    print("INTENT CLASSIFICATION")
    print("=" * 70)

    intent_true = df["expected_intent"]
    intent_pred = df["predicted_intent"]

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    intent_accuracy = accuracy_score(
        intent_true,
        intent_pred
    )

    print(
        f"\nAccuracy: {intent_accuracy:.2%}"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print("\nClassification Report:")

    print(
        classification_report(
            intent_true,
            intent_pred,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Intent confusion matrix
    # --------------------------------------------------------

    intent_labels = sorted(
        set(intent_true) | set(intent_pred)
    )

    print("Intent Confusion Matrix:")
    print(
        confusion_matrix(
            intent_true,
            intent_pred,
            labels=intent_labels
        )
    )

    print("\nIntent Labels:")
    print(intent_labels)

    # ========================================================
    # ESCALATION CLASSIFICATION
    # ========================================================

    print("\n\n" + "=" * 70)
    print("ESCALATION CLASSIFICATION")
    print("=" * 70)

    escalate_true = df["expected_escalate"]
    escalate_pred = df["predicted_escalate"]

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    escalation_accuracy = accuracy_score(
        escalate_true,
        escalate_pred
    )

    print(
        f"\nAccuracy: {escalation_accuracy:.2%}"
    )

    # --------------------------------------------------------
    # Precision / Recall / F1
    # --------------------------------------------------------

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            escalate_true,
            escalate_pred,
            labels=["no", "yes"],
            average="binary",
            pos_label="yes",
            zero_division=0
        )
    )

    print(
        f"Precision (escalate): {precision:.2%}"
    )

    print(
        f"Recall (escalate): {recall:.2%}"
    )

    print(
        f"F1 (escalate): {f1:.2%}"
    )

    # --------------------------------------------------------
    # Escalation confusion matrix
    # --------------------------------------------------------

    print("\nEscalation Confusion Matrix:")

    matrix = confusion_matrix(
        escalate_true,
        escalate_pred,
        labels=["no", "yes"]
    )

    print(matrix)

    print("\nMatrix format:")
    print("[[TN, FP],")
    print(" [FN, TP]]")

    # ========================================================
    # ESCALATION DISTRIBUTION
    # ========================================================

    print("\n\n" + "=" * 70)
    print("ESCALATION DISTRIBUTION")
    print("=" * 70)

    print("\nExpected:")
    print(
        df["expected_escalate"]
        .value_counts()
    )

    print("\nPredicted:")
    print(
        df["predicted_escalate"]
        .value_counts()
    )

    # ========================================================
    # INTENT DISTRIBUTION
    # ========================================================

    print("\n\n" + "=" * 70)
    print("INTENT DISTRIBUTION")
    print("=" * 70)

    print("\nExpected intents:")
    print(
        df["expected_intent"]
        .value_counts()
    )

    print("\nPredicted intents:")
    print(
        df["predicted_intent"]
        .value_counts()
    )

    # ========================================================
    # MOST COMMON INTENT ERRORS
    # ========================================================

    print("\n\n" + "=" * 70)
    print("MOST COMMON INTENT ERRORS")
    print("=" * 70)

    intent_errors = df[
        df["expected_intent"] != df["predicted_intent"]
    ]

    if len(intent_errors) == 0:

        print("\nNo intent classification errors.")

    else:

        error_counts = (
            intent_errors[
                [
                    "expected_intent",
                    "predicted_intent"
                ]
            ]
            .value_counts()
        )

        print(error_counts.head(15))

    # ========================================================
    # ESCALATION ERRORS
    # ========================================================

    print("\n\n" + "=" * 70)
    print("ESCALATION ERRORS")
    print("=" * 70)

    escalation_errors = df[
        df["expected_escalate"] != df["predicted_escalate"]
    ]

    print(
        f"\nTotal escalation errors: "
        f"{len(escalation_errors)}"
    )

    if len(escalation_errors) > 0:

        print("\nExamples:")

        display_columns = []

        if "customer_message" in df.columns:
            display_columns.append("customer_message")

        elif "message" in df.columns:
            display_columns.append("message")

        display_columns.extend(
            [
                "expected_escalate",
                "predicted_escalate"
            ]
        )

        print(
            escalation_errors[
                display_columns
            ].head(10).to_string(index=False)
        )

    # ========================================================
    # OVERALL SUMMARY
    # ========================================================

    print("\n\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(
        f"\nNumber of evaluation examples: "
        f"{len(df)}"
    )

    print(
        f"Intent accuracy: "
        f"{intent_accuracy:.2%}"
    )

    print(
        f"Escalation accuracy: "
        f"{escalation_accuracy:.2%}"
    )

    print(
        f"Escalation precision: "
        f"{precision:.2%}"
    )

    print(
        f"Escalation recall: "
        f"{recall:.2%}"
    )

    print(
        f"Escalation F1: "
        f"{f1:.2%}"
    )

    print("\n" + "=" * 70)
    print("EVALUATION FINISHED")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()