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


## Day 4: RAG Faithfulness and Retrieval Quality

Day 4 focused on evaluating whether a RAG system answers from retrieved evidence.

A RAG system has two main parts:

* **Retriever** — finds relevant documents or context.
* **Generator** — creates the final answer using that context.

### Key Metrics

**Faithfulness** checks whether the generated answer agrees with the retrieved context.

**Contextual Relevancy** checks whether the retrieved context is actually useful for answering the user’s question.

### Important Claim Types

Claims in an AI answer can be:

* **Supported** — confirmed by the retrieved context.
* **Contradicted** — conflicts with the retrieved context.
* **Unsupported** — not confirmed or contradicted by the context.

A major lesson was:

> No contradiction does not always mean an answer is fully grounded.

The built-in faithfulness metric detected direct contradictions well, but it did not always penalize extra unsupported claims. This showed why built-in metrics must be calibrated against the product’s quality requirements.

### Honest Uncertainty

When the retrieved context does not contain enough information, a trustworthy RAG system should admit that instead of guessing.

Example:

```text
The available information does not specify whether return shipping is free.
```

### RAG Diagnosis

| Faithfulness | Contextual Relevancy | Likely Result                                 |
| ------------ | -------------------- | --------------------------------------------- |
| High         | High                 | Healthy RAG behavior                          |
| High         | Low                  | Retriever failed, generator responded safely  |
| Low          | High                 | Good context, but generator misused it        |
| Low          | Low                  | Retrieval and generation may both have failed |

### Key Lesson

A strong RAG evaluation suite should identify whether the failure came from:

* The retriever
* The generator
* The source data
* The evaluator itself

Day 4 demonstrated that reliable AI testing requires more than checking the final answer. It requires tracing every claim back to evidence and diagnosing which system component caused the failure.
