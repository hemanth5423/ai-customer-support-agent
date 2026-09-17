# AI Customer Support Agent — Technical Report

## 1. Problem framing

I built an  customer-support agent over the Customer Support on Twitter (TWCS) dataset. The agent:

1. classifies an incoming customer message into a small Domain -specific intent taxonomy;
2. retrieves historically similar Support interactions and uses their responses as grounding evidence;
3. decides whether the case should be auto-handled or escalated, with an explicit reason.

### What “good” means

For this MVP, a good response is **grounded, useful, safe, and appropriately routed**. The agent should avoid inventing policies, refunds, timelines, troubleshooting facts, or account actions that are not supported by its evidence.

### What I did not build

I did not build a fully autonomous support system, account actions, payments/refunds, or direct customer-data access. The MVP drafts a response and routing decision; sensitive or low-evidence cases can be escalated.

---

## 2. Data and evaluation setup

The primary dataset is the Customer Support on Twitter (TWCS) dataset, using the AppleSupport slice. The repository contains a context-rich AppleSupport corpus and a 200-example   evaluation set with intent and escalation labels.

The corrected evaluator uses `golden_set_final.csv` and normalizes escalation labels. An earlier evaluation path used an intermediate file with incomplete labels and inconsistent escalation encodings; its apparent 1% intent result was therefore not a valid headline metric.

A separate **50-example human response-quality audit** was completed independently. Human raters scored groundedness, helpfulness, safety, and overall quality from 1–5.

---

## 3. Method

### Architecture

```plantuml
@startuml
actor Customer
rectangle "Support Agent" {
  component "Input / Context" as Input
  component "Intent Classifier
LLM + deterministic fallback" as Intent
  component "Semantic Retriever" as Retrieve
  database "AppleSupport historical
conversation index" as Hist
  component "Grounded Response
LLM or retrieved response" as Response
  component "Escalation Policy" as Esc
}
Customer --> Input
Input --> Intent
Input --> Retrieve
Hist --> Retrieve
Retrieve --> Response
Input --> Response
Input --> Esc
Retrieve --> Esc
Intent --> Esc
Response --> Customer : draft reply + evidence
Esc --> Customer : auto / human + reason
@enduml
```

### Retrieval

The retriever uses `sentence-transformers/all-MiniLM-L6-v2` and cosine similarity over historical customer messages. The top 3 examples are retained. Normalized embeddings make similarity search inexpensive.

### Intent classification

The taxonomy is:

- `account_access`
- `billing`
- `delivery`
- `product_issue`
- `cancellation`
- `connectivity`
- `technical_support`
- `information_request`
- `complaint`
- `other`

With an OpenAI key, the optional LLM path uses structured JSON output for intent classification. Without an API key, the repository remains runnable with the deterministic fallback.

### Response generation

The response generator receives the customer message and top historical examples. In local mode, the highest-ranked historical support response is used as the safe fallback. In LLM mode, the generator is constrained to the retrieved evidence.

### Escalation

The current policy escalates when:

- retrieval similarity is below 0.45;
- security/legal/financial-risk signals appear (for example fraud, hacking, stolen device, unauthorized or duplicate charges);
- the customer reports repeated or unresolved attempts.

The policy emits a reason alongside the yes/no decision.

---

## 4. Results and baselines

On the 200-example evaluation set:

| Metric | Result |
|---|---:|
| Intent accuracy | **32.5%** |
| Escalation accuracy | **87.0%** |
| Escalation positive recall | **5.6% (1/18)** |

Intent baselines use an 80/20 stratified split with `random_state=42`:

| Baseline | Intent accuracy |
|---|---:|
| Majority class | **27.5%** |
| TF-IDF + Logistic Regression | **35.0%** |
| Current deterministic agent | **32.5%** |

The current intent classifier therefore does **not** beat the simple TF-IDF baseline on this benchmark.

The escalation number also needs context. There are 182 `no` and 18 `yes` labels. An always-`no` baseline would therefore achieve **91.0% accuracy**. The agent's 87.0% accuracy is below that trivial baseline, and it correctly identifies only 1 of the 18 escalation cases. This is an important production limitation.

### Human response-quality audit

For 50 independently rated response examples:

| Human metric | Mean / 5 |
|---|---:|
| Groundedness | **3.58** |
| Helpfulness | **3.46** |
| Safety | **3.50** |
| Overall | **3.64** |

These are human ratings of response quality, not intent accuracy.

### LLM-as-judge

A separate Gemini LLM judge was run on **5 examples** because the available API quota limited the evaluation. Its means were:

| Judge metric | Mean / 5 |
|---|---:|
| Groundedness | **4.60** |
| Helpfulness | **4.40** |
| Safety | **5.00** |
| Escalation appropriateness | **5.00** |
| Overall | **4.40** |

The five LLM-judged examples do **not overlap** with the 50 human-audited examples, so a human-vs-LLM judge agreement statistic cannot be honestly estimated from this run. The repository's scoring script detects overlap rather than manufacturing agreement.

The older automated response-quality proxy marked 49/50 audited responses as `good` and 1/50 as `acceptable`. Its correlation with human overall scores was only **0.139**, so it is not suitable as a headline quality metric.

---

## 5. Top failure modes

1. **Overlapping intent labels** — account, product, connectivity, and technical issues can describe the same underlying device/software problem.
2. **Multi-intent messages** — one tweet may contain a technical problem plus a complaint or billing issue while the benchmark requires one label.
3. **Context ambiguity** — conversation history contains previous support responses and can introduce vocabulary unrelated to the current customer need.
4. **Semantic retrieval is not operational equivalence** — two messages can be semantically close while requiring different support actions.
5. **Escalation imbalance** — high overall escalation accuracy hides very poor recall for the minority `yes` class.

---

## 6. What is misleading about my headline number?

The most misleading numbers are the apparent **1% intent accuracy** from the earlier evaluator and the **87% escalation accuracy** when viewed without class balance.

The 1% figure came from an intermediate golden file with incomplete labels and an inconsistent escalation-label format. After correcting the evaluation path, intent accuracy is 32.5%.

The 87% escalation accuracy sounds strong, but the dataset contains only 18 escalation cases out of 200. The agent catches only 1 of those 18 cases, while an always-`no` policy would reach 91% accuracy. For a real support deployment, positive-class recall and risk-weighted error cost matter more than raw accuracy.

---

## 7. Evaluation harness

The repository provides:

- `src/evaluation/evaluate_corrected.py` — intent and escalation evaluation;
- `src/evaluation/baselines_corrected.py` — majority and TF-IDF baselines;
- `src/evaluation/failure_analysis.py` — failure extraction;
- `src/evaluation/llm_judge.py` — Gemini LLM-as-judge;
- `src/evaluation/prepare_human_audit.py` — creates a response-quality audit;
- `src/evaluation/score_human_audit.py` — human metrics plus optional judge-overlap checks.

The human audit is stored in `data/evaluation/human_response_audit.csv`.

---

## 8. Next week

1. Double-label ambiguous intent cases with a written annotation guide.
2. Add a reranker over the top semantic-retrieval results.
3. Measure retrieval Recall@1/3/5 independently.
4. Calibrate escalation thresholds using a larger risk-labelled set and class-sensitive metrics.
5. Add adversarial tests for unsupported refunds, privacy requests, prompt injection, and low-evidence answers.
6. Measure latency and cost for local and LLM-enabled modes.
7. Add a regression suite for taxonomy, retrieval, response grounding, and escalation.

---

## 9. Reproduction

```bash
python -m pip install -r requirements.txt

# Corrected evaluation
python -m src.evaluation.evaluate_corrected --limit 200

# Baselines
python -m src.evaluation.baselines_corrected

# Human audit scoring
python -m src.evaluation.score_human_audit

# Optional Gemini LLM judge
# PowerShell:
$env:GEMINI_API_KEY="YOUR_KEY"
python -m src.evaluation.llm_judge --limit 5
```

The first run may download the Sentence Transformers model. Subsequent runs reuse the local cache.

