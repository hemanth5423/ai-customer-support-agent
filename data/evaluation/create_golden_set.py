import pandas as pd
import os

# Input: context-rich Apple Support conversations
input_file = "data/processed/apple_support_context.csv"

# Output: golden evaluation set
output_file = "data/evaluation/golden_set.csv"

# Number of evaluation examples
SAMPLE_SIZE = 200

print("Loading context-rich Apple Support data...")

df = pd.read_csv(input_file)

print(f"Total context-rich examples: {len(df)}")

# Remove rows without useful messages
df = df.dropna(subset=["customer_message", "support_response"])

# Randomly sample 200 examples
golden = df.sample(
    n=min(SAMPLE_SIZE, len(df)),
    random_state=42
).copy()

# Keep only the fields needed for evaluation
golden = golden[
    [
        "customer_tweet_id",
        "support_tweet_id",
        "customer_message",
        "support_response",
        "conversation_context"
    ]
]

# Add columns for human labels
golden["intent"] = ""
golden["escalate"] = ""

# Save
golden.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 60)
print("GOLDEN EVALUATION SET CREATED")
print("=" * 60)

print(f"Examples: {len(golden)}")
print(f"Saved to: {output_file}")

print("\nColumns:")
print(golden.columns.tolist())