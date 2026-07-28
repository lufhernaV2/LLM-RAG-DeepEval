from dotenv import load_dotenv

from deepeval.metrics import (
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

load_dotenv() 

def create_faithfulness_metric() -> FaithfulnessMetric:
    return FaithfulnessMetric(
        threshold=0.7,
        model="gpt-4.1-mini",
        include_reason=True,
    )

def create_contextual_relevancy_metric() -> ContextualRelevancyMetric:
    return ContextualRelevancyMetric(
        threshold=0.7,
        model="gpt-4.1-mini",
        include_reason=True,
    )

def create_strict_groundedness_metric() -> GEval:
    return GEval(
        name="Strict Groundedness",
        evaluation_steps=[
            "Extract every factual claim from the actual output.",
            "For each claim, determine whether the retrieval context explicitly supports it.",
            "Treat claims absent from the retrieval context as unsupported.",
            "Treat claims contradicting the retrieval context as unsupported.",
            "Do not use outside knowledge to validate a claim.",
            "Assign a high score only when every factual claim is supported by the retrieval context.",
            "Lower the score for each unsupported, invented, or contradictory claim.",
        ],
        evaluation_params=[
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.RETRIEVAL_CONTEXT,
        ],
        threshold=0.7,
        model="gpt-4.1-mini",
    )

def measure_strict_groundedness(test_case: LLMTestCase) -> float:
    metric = create_strict_groundedness_metric()
    metric.measure(test_case)

    assert metric.score is not None, (
        "Strict groundedness metric did not produce a score. "
        f"Reason: {metric.reason}"
    )

    print(
        "\nStrict Groundedness"
        f"\nScore: {metric.score}"
        f"\nReason: {metric.reason}"
    )

    return metric.score


def measure_contextual_relevancy(test_case: LLMTestCase) -> float:
    metric = create_contextual_relevancy_metric()
    metric.measure(test_case)

    assert metric.score is not None, (
        "Contextual relevancy metric did not produce a score. "
        f"Reason: {metric.reason}"
    )

    print(
        "\nContextual Relevancy"
        f"\nScore: {metric.score}"
        f"\nReason: {metric.reason}"
    )

    return metric.score

def test_irrelevant_retrieval_context_fails_relevancy() -> None:
    test_case = LLMTestCase(
        input="What is the return policy?",
        actual_output=(
            "The retrieved information does not describe the "
            "company's return policy."
        ),
        retrieval_context=[
            "The company was founded in 2018.",
            "The headquarters is located in Orlando, Florida.",
            "The company employs approximately 400 people.",
        ],
    )

    faithfulness_score = measure_faithfulness(test_case)
    contextual_relevancy_score = measure_contextual_relevancy(test_case)

    assert faithfulness_score >= 0.7, (
        "The answer faithfully admits that the retrieved information "
        "does not contain the requested policy."
    )

    assert contextual_relevancy_score < 0.7, (
        "The retriever returned information unrelated to the return policy."
    )

def measure_faithfulness(test_case: LLMTestCase) -> float:
    metric = create_faithfulness_metric()
    metric.measure(test_case)

    assert metric.score is not None, (
        "Faithfulness metric did not produce a score. "
        f"Reason: {metric.reason}"
    )

    print(
        "\nFaithfulness"
        f"\nScore: {metric.score}"
        f"\nReason: {metric.reason}"
    )

    return metric.score


def test_answer_contains_supported_and_unsupported_claims() -> None:
    test_case = LLMTestCase(
        input="What is the return policy?",
        actual_output=(
            "Unused products may be returned within 30 days of delivery. "
            "A receipt is required. The company provides free return shipping, "
            "issues refunds within two hours, and gives every customer a $25 credit."
        ),
        retrieval_context=[
            "Customers may return unused products within 30 days of delivery.",
            "A receipt is required for all returns.",
        ],
    )

    faithfulness_score = measure_faithfulness(test_case)
    groundedness_score = measure_strict_groundedness(test_case)

    assert faithfulness_score >= 0.7, (
        "The added claims do not directly contradict the retrieved facts."
    )

    assert groundedness_score < 0.7, (
        "Several factual claims are not supported by the retrieval context."
    )

def test_answer_contradicts_retrieval_context() -> None:
    test_case = LLMTestCase(
        input="What is the return policy?",
        actual_output=(
            "Unused products may be returned within 60 days of delivery, "
            "and no receipt is required."
        ),
        retrieval_context=[
            (
                "Customers may return unused products within 30 days "
                "of delivery."
            ),
            "A receipt is required for all returns.",
        ],
    )

    score = measure_faithfulness(test_case)

    assert score < 0.7, (
        "The answer contradicts both the return window and "
        "receipt requirement."
    )

def test_answer_contains_supported_and_unsupported_claims() -> None:
    test_case = LLMTestCase(
        input="What is the return policy?",
        actual_output=(
            "Unused products may be returned within 30 days of delivery. "
            "A receipt is required. The company provides free return shipping, "
            "issues refunds within two hours, and gives every customer a $25 credit."
        ),
        retrieval_context=[
            (
                "Customers may return unused products within 30 days "
                "of delivery."
            ),
            "A receipt is required for all returns.",
        ],
    )

    score = measure_faithfulness(test_case)

    assert score < 0.7, (
        "Several claims are unsupported by the retrieval context and "
        "should cause the answer to fail faithfulness."
    )

def test_answer_admits_context_is_insufficient() -> None:
    test_case = LLMTestCase(
        input="Is return shipping free?",
        actual_output=(
            "The provided information does not specify whether "
            "return shipping is free."
        ),
        retrieval_context=[
            (
                "Customers may return unused products within 30 days "
                "of delivery."
            ),
            "A receipt is required for all returns.",
        ],
    )

    score = measure_faithfulness(test_case)

    assert score >= 0.7, (
        "The answer correctly avoids inventing information that "
        "is absent from the retrieval context."
    )   

def test_irrelevant_context_with_honest_answer() -> None:
    test_case = LLMTestCase(
        input="What is the return policy?",
        actual_output=(
            "The retrieved information does not include the company's "
            "return policy."
        ),
        retrieval_context=[
            "The company was founded in 2018.",
            "Its headquarters is located in Orlando.",
            "The company employs approximately 400 people.",
        ],
    )

    faithfulness_score = measure_faithfulness(test_case)
    contextual_relevancy_score = measure_contextual_relevancy(test_case)

    assert faithfulness_score >= 0.7
    assert contextual_relevancy_score < 0.7