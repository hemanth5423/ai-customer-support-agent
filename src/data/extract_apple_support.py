import pandas as pd

input_file = "data/raw/twcs/twcs.csv"
output_file = "data/processed/apple_support.csv"

brand = "AppleSupport"

# STEP 1: Find all tweets written by AppleSupport
apple_tweet_ids = set()

for chunk in pd.read_csv(input_file, chunksize=100000):
    brand_rows = chunk[chunk["author_id"] == brand]
    apple_tweet_ids.update(brand_rows["tweet_id"].astype(str))

print(f"AppleSupport tweets found: {len(apple_tweet_ids)}")


# STEP 2: Extract AppleSupport tweets
# and customer tweets that directly reply to AppleSupport
first_chunk = True

for chunk in pd.read_csv(input_file, chunksize=100000):

    chunk["tweet_id"] = chunk["tweet_id"].astype(str)
    chunk["in_response_to_tweet_id"] = (
        chunk["in_response_to_tweet_id"].astype("string")
    )

    selected = chunk[
        (chunk["author_id"] == brand)
        |
        (chunk["in_response_to_tweet_id"].isin(apple_tweet_ids))
    ]

    selected.to_csv(
        output_file,
        mode="w" if first_chunk else "a",
        header=first_chunk,
        index=False
    )

    first_chunk = False

print("AppleSupport data extraction complete.")
print(f"Saved to: {output_file}")