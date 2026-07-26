## Day 1 Findings

I created an Answer Relevancy quality gate with DeepEval.

Results:

- A directly relevant response scored highly and passed.
- An unrelated response scored 0 and failed.
- A partially relevant but incomplete response still passed.

This demonstrated that answer relevancy does not measure every aspect of
response quality. A response can be related to the question without fully
answering it, so production evaluations require multiple complementary metrics.


## Day 2: Correctness and Completeness

I added custom G-Eval metrics for answer correctness and completeness.

Initial calibration showed that metric dimensions can overlap:

- The correctness metric penalized a true but incomplete response.
- The completeness metric penalized both missing information and factual errors.
- A materially incorrect explanation still received partial correctness credit.

I refined the evaluation criteria to isolate each dimension more clearly.
This demonstrated that LLM-based evaluators must themselves be tested and
calibrated before they can be trusted as release gates.