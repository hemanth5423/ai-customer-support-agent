from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FILE = ROOT / "data/evaluation/human_response_audit.csv"

df = pd.read_csv(FILE)
score_cols = ["human_grounded", "human_helpful", "human_safe", "human_overall"]
for c in score_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

rated = df.dropna(subset=score_cols)
print(f"Human-rated examples: {len(rated)}/{len(df)}")
if rated.empty:
    raise SystemExit("No completed human ratings found.")

print("Mean groundedness:", round(rated.human_grounded.mean(), 2))
print("Mean helpfulness:", round(rated.human_helpful.mean(), 2))
print("Mean safety:", round(rated.human_safe.mean(), 2))
print("Mean overall:", round(rated.human_overall.mean(), 2))

# The old automated proxy is intentionally reported only as a weak sanity check.
proxy = rated.response_quality.map({"good": 5, "acceptable": 3, "poor": 1})
comparable = rated.assign(proxy=proxy).dropna(subset=["proxy"])
if len(comparable) >= 2:
    corr = comparable[["proxy", "human_overall"]].corr().iloc[0, 1]
    print("Proxy-vs-human ordinal correlation:", round(corr, 3))
    print("Proxy agreement is not a substitute for LLM-judge/human agreement.")

# LLM-judge agreement is computed only when the same customer+response pairs exist.
judge_file = ROOT / "data/evaluation/llm_judge_results.csv"
if judge_file.exists():
    judge = pd.read_csv(judge_file)
    overlap = rated.merge(
        judge[["customer_message", "response", "overall"]],
        on=["customer_message", "response"],
        how="inner",
        suffixes=("_human", "_judge"),
    )
    print("Human/LLM-judge overlapping examples:", len(overlap))
    if len(overlap) >= 2:
        corr = overlap[["human_overall", "overall"]].corr().iloc[0, 1]
        print("Human-vs-LLM-judge ordinal correlation:", round(corr, 3))
    else:
        print("Not enough overlapping examples to estimate judge agreement.")
else:
    print("LLM judge results file not found.")
