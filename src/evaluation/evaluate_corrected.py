"""Leakage-safe evaluation for the hand-labelled 200-example set.

The old evaluation script accidentally read golden_set.csv, where most labels
were still blank, and compared y/n against yes/no in one path. This script
uses golden_set_final.csv and normalizes labels before scoring.
"""

from pathlib import Path
import sys
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from agent.agent import CustomerSupportAgent
from agent.semantic_retriever import SemanticRetriever


def norm_bool(v):
    s = str(v).strip().lower()
    return "yes" if s in {"y", "yes", "true", "1"} else "no"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()

    golden = pd.read_csv(ROOT / "data/evaluation/golden_set_final.csv")
    golden = golden.dropna(subset=["conversation_context", "intent", "escalate"])
    golden = golden[golden.intent.ne("skipped")].head(args.limit).copy()

    retriever = SemanticRetriever(
        str(ROOT / "data/processed/apple_support_context.csv"), top_k=3
    )
    agent = CustomerSupportAgent(retriever)

    rows = []
    for _, row in golden.iterrows():
        result = agent.handle(str(row["conversation_context"]))
        rows.append({
            "customer_message": row["conversation_context"],
            "expected_intent": str(row["intent"]).strip(),
            "expected_escalate": norm_bool(row["escalate"]),
            "predicted_intent": result["intent"],
            "predicted_escalate": norm_bool(result["escalate"]),
            "response": result["response"],
            "reason": result.get("reason", ""),
            "similarity": result["similarity"],
        })

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "data/evaluation/evaluation_results_corrected.csv", index=False)

    intent_acc = accuracy_score(out.expected_intent, out.predicted_intent)
    esc_acc = accuracy_score(out.expected_escalate, out.predicted_escalate)
    print(f"Examples: {len(out)}")
    print(f"Intent accuracy: {intent_acc:.3f}")
    print(f"Escalation accuracy: {esc_acc:.3f}")
    print("\nIntent report:")
    print(classification_report(out.expected_intent, out.predicted_intent, zero_division=0))
    print("\nEscalation confusion matrix [no, yes]:")
    print(confusion_matrix(out.expected_escalate, out.predicted_escalate, labels=["no", "yes"]))


if __name__ == "__main__":
    main()
