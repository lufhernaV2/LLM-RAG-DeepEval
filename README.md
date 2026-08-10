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


## Day 5 – Retrieval Precision, Recall, and Ranking Quality

Day 5 focused on diagnosing retriever quality in RAG systems using DeepEval.

### Metrics

- **Contextual Recall** checks whether the retrieved context contains all evidence needed to support the expected answer.
- **Contextual Relevancy** checks whether the retrieved content is relevant to the user’s question.
- **Contextual Precision** checks whether relevant chunks are ranked above irrelevant chunks.

### Controlled Cases

Tested retrieval scenarios with:

- Complete and relevant evidence
- Missing required evidence
- Relevant evidence mixed with noise
- Relevant chunks ranked below irrelevant chunks
- Partially useful retrieval containing both missing evidence and noise

### Key Lessons

- One relevant chunk does not prove that retrieval is complete.
- Missing evidence primarily lowers contextual recall.
- Irrelevant chunks primarily lower contextual relevancy.
- Poor chunk ordering lowers contextual precision.
- A retriever may pass precision while failing recall and relevancy.
- `expected_output` should contain atomic, independently testable facts.
- Unexpected evaluator behavior should be treated as a calibration finding.
- Multiple retrieval metrics are needed to identify the true root cause of a RAG failure.


## Day 6 – End-to-End RAG Evaluation

Day 6 focused on diagnosing whether RAG failures came from the retriever or the generator.

### Metrics Used

* `AnswerRelevancyMetric` – checks whether the answer addresses the question.
* `FaithfulnessMetric` – checks whether the answer aligns with retrieved evidence.
* `ContextualRecallMetric` – checks whether required evidence is missing.
* `ContextualRelevancyMetric` – checks for irrelevant retrieval noise.
* `ContextualPrecisionMetric` – checks whether useful chunks are ranked highly.

### Key Findings

```text
Low recall + high faithfulness
→ Missing retrieval evidence, but the generator stayed grounded.

High recall + low faithfulness
→ Retrieval succeeded, but the generator misused the evidence.

Low relevancy
→ Too much retrieval noise.

Low precision
→ Relevant chunks were ranked too low.
```

A custom deterministic `BaseMetric` was also created to fail unsupported material policy claims. Known aliases such as `restocking fee`, `return fee`, and `return charge` were used to improve coverage.

### Main Lesson

Use LLM-based metrics for semantic diagnosis and deterministic checks for strict business rules and CI gates.


## Day 7 – Practical Evaluation Datasets

Day 7 focused on organizing reusable RAG test scenarios with DeepEval Goldens and EvaluationDataset.

Key Concepts
A Golden stores stable test data such as the input, expected output, and metadata.
An EvaluationDataset groups multiple Goldens into a reusable regression suite.
An LLMTestCase combines Golden data with runtime values such as actual_output and retrieval_context.
Practical Flow
Golden
+ runtime answer
+ runtime retrieval context
= LLMTestCase
What Was Built
A Northstar policy dataset with three Goldens.
Metadata for category and risk.
Dataset integrity tests.
Validation for required Golden fields.
A test that converts a Golden into an LLMTestCase.
Main Lesson

Evaluation datasets let the same important scenarios be rerun after prompt, model, retriever, or application changes. This creates a repeatable AI regression-testing foundation.

## Day 8 – Reusable Dataset Evaluation Runner

Day 8 focused on converting reusable Goldens into completed `LLMTestCase` objects.

### What Was Built

- A simulated RAG application that returns an answer and retrieval context.
- A `RAGResult` dataclass for storing runtime results.
- A reusable function that converts a Golden into an `LLMTestCase`.
- A dataset runner that processes every Golden automatically.
- Tests confirming that runtime fields and metadata are preserved.

### Practical Flow

```text
Golden
→ Run RAG application
→ Capture answer and retrieved chunks
→ Build LLMTestCase
→ Run evaluations

## Day 9 – Batch Evaluation and Regression Detection

Day 9 focused on running multiple RAG test cases through a reusable DeepEval metric pipeline.

### What Was Built

- Batch evaluation using `evaluate()`
- `AnswerRelevancyMetric`
- `FaithfulnessMetric`
- Sequential metric execution for easier debugging
- A controlled regression test

### Results

**Healthy baseline**

- 3 tests
- 100% pass rate

**Controlled regression**

- Answer Relevancy: `1.0` — PASS
- Faithfulness: `0.50` — FAIL
- Overall pass rate: `66.67%`

### Key Lessons

- Batch evaluation allows an entire Golden dataset to be tested consistently.
- A response can be highly relevant while still being unfaithful to retrieved evidence.
- Aggregate averages can hide individual critical failures.
- A useful regression suite should confirm that good behavior passes and known bad behavior fails.


## Day 10 – AI Quality Gates

### What I Learned

* Used DeepEval's `assert_test()` to turn evaluation metrics into automated pass/fail quality gates.
* Learned the difference between **measuring AI quality** and **enforcing AI quality requirements**.
* Created a Faithfulness gate with a minimum threshold of `0.7`.
* Verified that a known-bad response correctly fails the quality gate.
* Verified that a supported response correctly passes the quality gate.
* Connected the existing Golden dataset and RAG evaluation runner to parametrized pytest tests.
* Added multiple requirements to the same quality gate:

  * Answer Relevancy
  * Faithfulness
* Confirmed that every Golden must pass individually instead of relying only on aggregate metric averages.
* Created a controlled regression where:

  * Answer Relevancy = `1.0`
  * Faithfulness = `0.5`
  * Overall test = `FAILED`
* Learned that a response can directly answer the user's question while still contradicting retrieved evidence.
* Separated **release quality gates** from **metric calibration tests**.

### Key Takeaway

AI evaluation should not stop at reporting scores.

A practical AI QA framework should enforce minimum quality requirements so that critical regressions can automatically fail a test before release.

Current evaluation flow:

Golden → RAG Application → LLMTestCase → Metrics → Quality Gate → PASS / FAIL
