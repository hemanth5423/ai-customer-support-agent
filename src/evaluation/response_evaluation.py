from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from agent.agent import CustomerSupportAgent
from agent.semantic_retriever import SemanticRetriever

GOLDEN_FILE = ROOT / "data/evaluation/golden_set_final.csv"
CONTEXT_FILE = ROOT / "data/processed/apple_support_context.csv"
OUTPUT_FILE = ROOT / "data/evaluation/response_evaluation.csv"


def norm_bool(v):
    return "yes" if str(v).strip().lower() in {"y", "yes", "true", "1"} else "no"


def proxy_quality(response, similarity):
    response = str(response).strip()
    if not response:
        return "poor"
    if similarity < 0.45:
        return "poor"
    if similarity >= 0.55 and len(response.split()) >= 5:
        return "good"
    return "acceptable"


def main():
    df = pd.read_csv(GOLDEN_FILE)
    df = df.dropna(subset=["conversation_context", "intent", "escalate"])
    df = df[df.intent.ne("skipped")].copy()

    retriever = SemanticRetriever(str(CONTEXT_FILE), top_k=3)
    agent = CustomerSupportAgent(retriever)

    rows = []
    for _, row in df.iterrows():
        result = agent.handle(str(row["conversation_context"]))
        rows.append({
            "customer_message": row["conversation_context"],
            "expected_intent": row["intent"],
            "expected_escalate": norm_bool(row["escalate"]),
            "predicted_intent": result["intent"],
            "predicted_escalate": norm_bool(result["escalate"]),
            "response": result["response"],
            "reason": result.get("reason", ""),
            "similarity": result["similarity"],
            "response_quality": proxy_quality(result["response"], result["similarity"]),
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_FILE, index=False)
    print(f"Examples evaluated: {len(out)}")
    print(f"Intent accuracy: {(out.expected_intent == out.predicted_intent).mean():.3f}")
    print(f"Escalation accuracy: {(out.expected_escalate == out.predicted_escalate).mean():.3f}")
    print(out.response_quality.value_counts())
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
