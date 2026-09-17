# Decision Log

1. **Selected Support account** — it has enough recurring support interactions to build a useful retrieval corpus and is directly represented in the TWCS data.
2. **Used context-rich conversations rather than isolated replies** — previous turns provide important clues about what the customer already tried.
3. **Kept the taxonomy small** — a small set is easier to evaluate and less ambiguous than trying to reproduce every  product category.
4. **Chose semantic retrieval** — support requests use slang, spelling errors and paraphrases, making lexical matching brittle.
5. **Kept top-k at 3** — enough evidence for response drafting without giving the generator a large, noisy context.
6. **Used retrieved historical responses as the non-LLM fallback** — this guarantees that the baseline reply is grounded in observed support behaviour.
7. **Added a deterministic escalation policy** — routing should remain auditable and should not depend solely on free-form generation.
8. **Escalate low-similarity cases** — the agent should not confidently answer when the evidence base has no close analogue.
9. **Added an optional LLM path instead of making the API mandatory** — the repository remains runnable without credentials while supporting a stronger classifier/generator.
10. **Used structured output for LLM classification/judging** — downstream code needs a fixed schema rather than parsing arbitrary prose.
11. **Separated agent evaluation from response-quality judging** — a model that classifies correctly can still produce a poor answer.
12. **Created a human-audit sample** — automated judges need empirical agreement evidence; their scores should not be treated as ground truth.
13. **Validated and corrected the evaluation source before interpreting metrics** — the original script included unlabeled rows and inconsistent escalation encodings, making its 1% headline misleading.
14. **Did not hide the taxonomy ambiguity** — overlapping labels and multi-intent messages are genuine limitations of the benchmark and should be reported rather than tuned away on the test set.
