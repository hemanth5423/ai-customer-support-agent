from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data/evaluation/response_evaluation.csv"
OUTPUT = ROOT / "data/evaluation/human_response_audit.csv"

df = pd.read_csv(INPUT)
sample = df.sample(n=min(50, len(df)), random_state=42).copy()
sample["human_grounded"] = ""
sample["human_helpful"] = ""
sample["human_safe"] = ""
sample["human_overall"] = ""
sample["human_notes"] = ""
sample.to_csv(OUTPUT, index=False)
print(f"Created {len(sample)} human-audit rows: {OUTPUT}")
print("Rate each response 1-5 for groundedness, helpfulness and safety; overall is 1-5.")
