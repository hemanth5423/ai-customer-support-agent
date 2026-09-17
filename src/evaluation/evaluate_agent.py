from pathlib import Path
import sys
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

GOLDEN_FILE = ROOT / "data" / "evaluation" / "golden_set.csv"

OUTPUT_FILE = (
    ROOT
    / "data"
    / "evaluation"
    / "evaluation_results.csv"
)


# ============================================================
# IMPORT AGENT
# ============================================================

sys.path.insert(0, str(ROOT / "src"))

from agent.agent import CustomerSupportAgent
from agent.semantic_retriever import SemanticRetriever


# ============================================================
# NORMALIZE ESCALATION LABEL
# ============================================================

def normalize_escalation(value):

    value = str(value).strip().lower()

    if value in ["y", "yes", "true", "1"]:
        return "yes"

    return "no"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AI CUSTOMER SUPPORT AGENT - EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD GOLDEN SET
    # --------------------------------------------------------

    print("\nLoading golden set...")

    golden = pd.read_csv(GOLDEN_FILE)

    print(
        f"Loaded {len(golden)} evaluation examples."
    )

    # --------------------------------------------------------
    # LOAD RETRIEVER
    # --------------------------------------------------------

    context_file = (
        ROOT
        / "data"
        / "processed"
        / "apple_support_context.csv"
    )

    print("\nLoading semantic retriever...")

    retriever = SemanticRetriever(
        str(context_file),
        top_k=3
    )

    agent = CustomerSupportAgent(retriever)

    # --------------------------------------------------------
    # RUN AGENT
    # --------------------------------------------------------

    results = []

    print("\nRunning evaluation...")

    for index, row in golden.iterrows():

        customer_message = str(
            row["customer_message"]
        )

        result = agent.handle(
            customer_message
        )

        results.append({

            # Original customer question
            "customer_message":
                customer_message,

            # Expected labels
            "expected_intent":
                row["intent"],

            "expected_escalate":
                normalize_escalation(
                    row["escalate"]
                ),

            # Agent predictions
            "predicted_intent":
                result["intent"],

            "predicted_escalate":
                normalize_escalation(
                    result["escalate"]
                ),

            # Generated/retrieved response
            "response":
                result["response"],

            # Retrieval confidence
            "similarity":
                result["similarity"]
        })

        if (index + 1) % 25 == 0:

            print(
                f"Processed "
                f"{index + 1}/{len(golden)}"
            )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # INTENT EVALUATION
    # ========================================================

    print("\n")
    print("=" * 70)
    print("INTENT CLASSIFICATION")
    print("=" * 70)

    intent_accuracy = accuracy_score(
        results_df["expected_intent"],
        results_df["predicted_intent"]
    )

    print(
        f"\nIntent Accuracy: "
        f"{intent_accuracy:.3f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            results_df["expected_intent"],
            results_df["predicted_intent"],
            zero_division=0
        )
    )

    # ========================================================
    # ESCALATION EVALUATION
    # ========================================================

    print("\n")
    print("=" * 70)
    print("ESCALATION")
    print("=" * 70)

    escalation_accuracy = accuracy_score(
        results_df["expected_escalate"],
        results_df["predicted_escalate"]
    )

    print(
        f"\nEscalation Accuracy: "
        f"{escalation_accuracy:.3f}"
    )

    print("\nEscalation Report:")

    print(
        classification_report(
            results_df["expected_escalate"],
            results_df["predicted_escalate"],
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # ESCALATION CONFUSION MATRIX
    # --------------------------------------------------------

    print("\nEscalation Confusion Matrix:")

    print(
        pd.crosstab(
            results_df["expected_escalate"],
            results_df["predicted_escalate"]
        )
    )

    # ========================================================
    # RESULTS LOCATION
    # ========================================================

    print("\n")
    print("=" * 70)

    print(
        "Detailed results saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print("\n")
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()