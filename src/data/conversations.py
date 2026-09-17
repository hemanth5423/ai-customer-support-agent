import pandas as pd
from pathlib import Path

RAW_FILE = "data/raw/twcs/twcs.csv"
OUTPUT_FILE = "data/processed/apple_support_context.csv"

APPLE = "AppleSupport"

# ---------------------------------------------------------
# STEP 1: Collect AppleSupport tweets
# ---------------------------------------------------------

print("Step 1: Finding AppleSupport tweets...")

apple_ids = set()

for chunk in pd.read_csv(
    RAW_FILE,
    chunksize=100000,
    dtype=str,
    usecols=[
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "in_response_to_tweet_id"
    ]
):

    apple_rows = chunk[chunk["author_id"] == APPLE]

    apple_ids.update(
        apple_rows["tweet_id"].dropna()
    )

print(f"AppleSupport tweets found: {len(apple_ids)}")


# ---------------------------------------------------------
# STEP 2: Find tweets connected to AppleSupport
# ---------------------------------------------------------

print("\nStep 2: Collecting conversation history...")

needed_ids = set(apple_ids)

# We need the parent of every AppleSupport tweet.
for chunk in pd.read_csv(
    RAW_FILE,
    chunksize=100000,
    dtype=str,
    usecols=[
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "in_response_to_tweet_id"
    ]
):

    rows = chunk[chunk["tweet_id"].isin(needed_ids)]

    parents = rows["in_response_to_tweet_id"].dropna()

    needed_ids.update(parents)

print(f"Initial conversation tweets needed: {len(needed_ids)}")


# ---------------------------------------------------------
# STEP 3: Repeatedly find previous messages
# ---------------------------------------------------------

previous_count = 0

while True:

    found = {}

    for chunk in pd.read_csv(
        RAW_FILE,
        chunksize=100000,
        dtype=str,
        usecols=[
            "tweet_id",
            "author_id",
            "inbound",
            "created_at",
            "text",
            "in_response_to_tweet_id"
        ]
    ):

        rows = chunk[chunk["tweet_id"].isin(needed_ids)]

        for _, row in rows.iterrows():
            found[row["tweet_id"]] = row.to_dict()

    new_parents = set()

    for row in found.values():

        parent = row.get("in_response_to_tweet_id")

        if pd.notna(parent) and parent not in needed_ids:
            new_parents.add(parent)

    needed_ids.update(new_parents)

    print(f"Found {len(found)} tweets, added {len(new_parents)} previous messages.")

    if len(new_parents) == 0:
        break

    if len(needed_ids) > 1000000:
        print("Conversation history limit reached.")
        break


# ---------------------------------------------------------
# STEP 4: Build tweet lookup
# ---------------------------------------------------------

print("\nStep 3: Building tweet lookup...")

tweets = {}

for chunk in pd.read_csv(
    RAW_FILE,
    chunksize=100000,
    dtype=str,
    usecols=[
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "in_response_to_tweet_id"
    ]
):

    rows = chunk[chunk["tweet_id"].isin(needed_ids)]

    for _, row in rows.iterrows():

        tweets[row["tweet_id"]] = {
            "tweet_id": row["tweet_id"],
            "author_id": row["author_id"],
            "inbound": row["inbound"],
            "created_at": row["created_at"],
            "text": row["text"],
            "parent": row["in_response_to_tweet_id"]
        }


# ---------------------------------------------------------
# STEP 5: Create context-rich examples
# ---------------------------------------------------------

print("\nStep 4: Building context-rich examples...")

examples = []

for apple_id in apple_ids:

    if apple_id not in tweets:
        continue

    apple_tweet = tweets[apple_id]

    parent_id = apple_tweet["parent"]

    if pd.isna(parent_id) or parent_id not in tweets:
        continue

    customer = tweets[parent_id]

    # Build previous conversation
    history = []

    current_id = parent_id

    while current_id in tweets:

        current = tweets[current_id]

        history.append(
            current["text"]
        )

        current_id = current["parent"]

        if pd.isna(current_id):
            break

        if len(history) >= 5:
            break

    history.reverse()

    # Need at least 2 messages of context
    if len(history) < 2:
        continue

    context = "\n".join(history[:-1])

    examples.append({
        "customer_tweet_id": customer["tweet_id"],
        "support_tweet_id": apple_id,
        "customer_message": customer["text"],
        "support_response": apple_tweet["text"],
        "conversation_context": context
    })


# ---------------------------------------------------------
# STEP 6: Save
# ---------------------------------------------------------

output = pd.DataFrame(examples)

Path(OUTPUT_FILE).parent.mkdir(
    parents=True,
    exist_ok=True
)

output.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 60)
print("CONTEXT EXTRACTION COMPLETE")
print("=" * 60)

print(f"Context-rich examples: {len(output)}")
print(f"Saved to: {OUTPUT_FILE}")

print("\nColumns:")
print(output.columns.tolist())