import pandas as pd
from sentence_transformers import SentenceTransformer, util


# --------------------------------------------------
# 1. LOAD GOLDEN SET
# --------------------------------------------------

INPUT_FILE = "data/evaluation/golden_set.csv"
OUTPUT_FILE = "data/evaluation/golden_set_suggestions.csv"

df = pd.read_csv(INPUT_FILE)

print(f"Loaded {len(df)} evaluation examples.")


# --------------------------------------------------
# 2. OUR 10 INTENT CATEGORIES
# --------------------------------------------------

intent_descriptions = {
    "account_access": "Problems logging in, forgotten passwords, Apple ID, account access, verification codes",
    "billing": "Payment problems, charges, refunds, invoices, subscriptions, duplicate charges",
    "delivery": "Delivery, shipping, tracking, missing package, delayed package",
    "product_issue": "Product not working, defective product, damaged product, feature not working",
    "cancellation": "Canceling an order, subscription, booking, or service",
    "connectivity": "Internet, network, mobile data, WiFi, signal, connection problems",
    "technical_support": "Technical troubleshooting, device problems, software problems, errors",
    "information_request": "Customer asking for information, instructions, availability, specifications",
    "complaint": "Customer expressing dissatisfaction, frustration, or a serious complaint",
    "other": "Anything that does not clearly fit the other categories"
}


# --------------------------------------------------
# 3. LOAD SEMANTIC MODEL
# --------------------------------------------------

print("Loading AI model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

intent_names = list(intent_descriptions.keys())
intent_texts = list(intent_descriptions.values())

intent_embeddings = model.encode(
    intent_texts,
    convert_to_tensor=True
)


# --------------------------------------------------
# 4. GENERATE INTENT SUGGESTIONS
# --------------------------------------------------

print("Generating intent suggestions...")

texts = df["conversation_context"].fillna("").tolist()

text_embeddings = model.encode(
    texts,
    convert_to_tensor=True,
    show_progress_bar=True
)

similarities = util.cos_sim(
    text_embeddings,
    intent_embeddings
)

best_scores, best_indices = similarities.max(dim=1)

df["suggested_intent"] = [
    intent_names[index]
    for index in best_indices.cpu().tolist()
]

df["intent_confidence"] = [
    round(float(score), 3)
    for score in best_scores.cpu().tolist()
]


# --------------------------------------------------
# 5. SIMPLE ESCALATION SUGGESTION
# --------------------------------------------------

def suggest_escalation(text):
    text = str(text).lower()

    escalation_words = [
        "fraud",
        "hacked",
        "stolen",
        "charged twice",
        "refund",
        "lawsuit",
        "legal",
        "security",
        "account compromised",
        "cannot access",
        "urgent",
        "angry",
        "complaint"
    ]

    for word in escalation_words:
        if word in text:
            return "yes"

    return "no"


df["suggested_escalate"] = df["conversation_context"].apply(
    suggest_escalation
)


# --------------------------------------------------
# 6. SAVE RESULTS
# --------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("=" * 60)
print("AUTO-LABELING COMPLETE")
print("=" * 60)

print(f"Examples processed: {len(df)}")
print(f"Output file: {OUTPUT_FILE}")

print()
print(df[
    [
        "suggested_intent",
        "intent_confidence",
        "suggested_escalate"
    ]
].head(10))