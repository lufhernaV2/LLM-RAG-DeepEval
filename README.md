## Day 1 Findings

I created an Answer Relevancy quality gate with DeepEval.

Results:

- A directly relevant response scored highly and passed.
- An unrelated response scored 0 and failed.
- A partially relevant but incomplete response still passed.

This demonstrated that answer relevancy does not measure every aspect of
response quality. A response can be related to the question without fully
answering it, so production evaluations require multiple complementary metrics.