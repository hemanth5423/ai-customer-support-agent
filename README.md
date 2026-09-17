# AI Customer Support Agent

A reproducible MVP for an AI customer-support agent grounded in historical customer-service conversations.

## What it does

- classifies a customer-support message;
- retrieves historically similar customer-support conversations;
- drafts a grounded response;
- decides `yes/no` escalation and records a reason;
- evaluates intent, escalation and response quality.

## Headline evaluation

On the 200-example corrected evaluation set:

- **Intent accuracy:** 32.5%
- **Escalation accuracy:** 87.0%
- **Escalation positive recall:** 5.6% (1/18)
- **Majority intent baseline:** 27.5%
- **TF-IDF + Logistic Regression baseline:** 35.0%
- - Always-no escalation baseline: 91.0%

A 50-example human response audit produced means of **3.58 groundedness, 3.46 helpfulness, 3.50 safety, and 3.64 overall** on a 1–5 scale.

A Gemini LLM judge was run on 5 examples because of free-tier quota limits: **4.60 groundedness, 4.40 helpfulness, 5.00 safety, 5.00 escalation appropriateness, 4.40 overall**. Those five examples do not overlap with the human audit, so no human-vs-LLM agreement statistic is claimed.

## Repository structure

```text
src/agent/              agent + retrieval + taxonomy
src/data/               dataset extraction/review helpers
src/evaluation/         evaluation, baselines, failure analysis, LLM judge
data/processed/         AppleSupport context corpus
data/evaluation/        golden set and evaluation outputs
REPORT.md               assignment report
DECISION_LOG.md         non-obvious design decisions
requirements.txt
```

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

python -m pip install -r requirements.txt
```

The semantic retriever uses `sentence-transformers/all-MiniLM-L6-v2`.

## Run the agent

```bash
python -m src.agent.agent
```

Without an API key, the agent uses the local deterministic intent classifier and historical-response fallback.

The optional OpenAI path can be enabled with `OPENAI_API_KEY`; it uses structured JSON output for intent classification.

## Evaluation

```bash
python -m src.evaluation.evaluate_corrected --limit 200
python -m src.evaluation.baselines_corrected
python -m src.evaluation.failure_analysis
```

The canonical evaluation entry point is `evaluate_corrected.py`. Do not use the older `evaluation_results.csv` as the headline metric without checking provenance.

## Human response-quality audit

The completed audit is stored in:

```text
data/evaluation/human_response_audit.csv
```

To score it:

```bash
python -m src.evaluation.score_human_audit
```

The script reports human means and only computes human-vs-LLM agreement when the exact customer-message/response pair exists in both datasets.

## LLM-as-judge

The repository uses the Gemini SDK for a separate response-quality judge.

PowerShell:

```powershell
$env:GEMINI_API_KEY="YOUR_KEY"
python -m src.evaluation.llm_judge --limit 5
```

The judge evaluates groundedness, helpfulness, safety, escalation appropriateness, and overall quality from 1–5. Results are saved incrementally to `data/evaluation/llm_judge_results.csv`.

The checked-in five-example judge run is intentionally described as a small smoke-test because free-tier quota limited the sample. Do not interpret it as a statistically robust quality estimate.

## Dataset

The project uses the Customer Support on Twitter (TWCS) dataset and the AppleSupport brand slice. The repository contains the processed AppleSupport context corpus and evaluation sample needed for the included runs; the full raw dataset is not required for the quick evaluation.

## Important limitation

The intent benchmark is difficult and contains overlapping/multi-intent cases. The current agent does not beat the TF-IDF baseline. Escalation accuracy is also misleading without class balance: only 18/200 examples require escalation, so an always-`no` baseline reaches 91% accuracy while the agent has only 1/18 positive recall.

See `REPORT.md` for the full analysis, failure modes, and next steps.
