from typing import Any

from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)
from deepeval.test_case import LLMTestCase

EVALUATOR_MODEL = "gpt-4.1-mini"
PASSING_THRESHOLD = 0.7

def create_contextual_precision_metric() -> ContextualPrecisionMetric:
    return ContextualPrecisionMetric(
        threshold=PASSING_THRESHOLD,
        model=EVALUATOR_MODEL,
        include_reason=True,
    )


def create_contextual_recall_metric() -> ContextualRecallMetric:
    return ContextualRecallMetric(
        threshold=PASSING_THRESHOLD,
        model=EVALUATOR_MODEL,
        include_reason=True,
        verbose_mode=True,
    )


def create_contextual_relevancy_metric() -> ContextualRelevancyMetric:
    return ContextualRelevancyMetric(
        threshold=PASSING_THRESHOLD,
        model=EVALUATOR_MODEL,
        include_reason=True,
    )

def measure_and_print(
    metric_name: str,
    metric: Any,
    test_case: LLMTestCase,
) -> None:
    metric.measure(test_case)

    print(f"\n{metric_name}")
    print(f"Score: {metric.score}")
    print(f"Threshold: {metric.threshold}")
    passed = metric.score >= metric.threshold
    print(f"Passed: {passed}")
    print(f"Reason: {metric.reason}")

def test_relevant_evidence_missing() -> None:
    test_case = LLMTestCase(
        input=(
            "I opened a laptop 20 days ago. Can I return it, "
            "is there a fee, and how will I receive my refund?"
        ),
        actual_output=(
            "Yes. Opened laptops may be returned within 30 days, "
            "and the refund will be issued to your original "
            "payment method."
        ),
        expected_output=(
            "Opened laptops may be returned within 30 days. "
            "Opened laptop returns are subject to a 15% restocking fee. "
            "Approved refunds are issued to the original payment method."
        ),
        retrieval_context=[
            (
                "Northstar Electronics accepts opened laptop returns "
                "within 30 calendar days of delivery."
            ),
            (
                "Approved refunds are issued to the customer's "
                "original payment method."
            ),
        ],
    )

    measure_and_print(
        metric_name="Contextual Recall",
        metric=create_contextual_recall_metric(),
        test_case=test_case,
    )

    measure_and_print(
        metric_name="Contextual Recall",
        metric=create_contextual_recall_metric(),
        test_case=test_case,
    )


def test_all_relevant_chunks_retrieved() -> None:
    test_case = LLMTestCase(
        input=(
            "I opened a laptop 20 days ago. Can I return it, "
            "is there a fee, and how will I receive my refund?"
        ),
        actual_output=(
            "Yes. Opened laptops may be returned within 30 days. "
            "A 15% restocking fee applies, and the refund will be "
            "issued to your original payment method."
        ),
        expected_output=(
            "Opened laptops may be returned within 30 days. "
            "A 15% restocking fee applies, and approved refunds "
            "are issued to the original payment method."
        ),
        retrieval_context=[
            (
                "Northstar Electronics accepts opened laptop returns "
                "within 30 calendar days of delivery."
            ),
            (
                "Opened electronics returned in otherwise eligible "
                "condition are subject to a 15% restocking fee."
            ),
            (
                "Approved refunds are issued to the customer's "
                "original payment method."
            ),
        ],
    )

    measure_and_print(
        metric_name="Contextual Relevancy",
        metric=create_contextual_relevancy_metric(),
        test_case=test_case,
    )

def test_relevant_chunks_ranked_too_low() -> None:
    test_case = LLMTestCase(
        input=(
            "I opened a laptop 20 days ago. Can I return it, "
            "is there a fee, and how will I receive my refund?"
        ),
        actual_output=(
            "Yes. Opened laptops may be returned within 30 days. "
            "A 15% restocking fee applies, and the refund will be "
            "issued to your original payment method."
        ),
        expected_output=(
            "Opened laptops may be returned within 30 days. "
            "Opened laptop returns are subject to a 15% restocking fee. "
            "Approved refunds are issued to the original payment method."
        ),
        retrieval_context=[
            (
                "Northstar Electronics was founded in 2014 and operates "
                "retail stores across several states."
            ),
            (
                "Employees must return company-issued laptops when "
                "their employment ends."
            ),
            (
                "Northstar Electronics accepts opened laptop returns "
                "within 30 calendar days of delivery."
            ),
            (
                "Opened electronics returned in otherwise eligible "
                "condition are subject to a 15% restocking fee."
            ),
            (
                "Approved refunds are issued to the customer's "
                "original payment method."
            ),
        ],
    )

    measure_and_print(
        metric_name="Contextual Precision",
        metric=create_contextual_precision_metric(),
        test_case=test_case,
    )

def test_relevant_evidence_mixed_with_noise() -> None:
    test_case = LLMTestCase(
        input=(
            "I opened a laptop 20 days ago. Can I return it, "
            "is there a fee, and how will I receive my refund?"
        ),
        actual_output=(
            "Yes. Opened laptops may be returned within 30 days. "
            "A 15% restocking fee applies, and the refund will be "
            "issued to your original payment method."
        ),
        expected_output=(
            "Opened laptops may be returned within 30 days. "
            "Opened laptop returns are subject to a 15% restocking fee. "
            "Approved refunds are issued to the original payment method."
        ),
        retrieval_context=[
            (
                "Northstar Electronics accepts opened laptop returns "
                "within 30 calendar days of delivery."
            ),
            (
                "Northstar Electronics was founded in 2014 and operates "
                "retail stores across several states."
            ),
            (
                "Opened electronics returned in otherwise eligible "
                "condition are subject to a 15% restocking fee."
            ),
            (
                "Employees must return company-issued laptops when "
                "their employment ends."
            ),
            (
                "Approved refunds are issued to the customer's "
                "original payment method."
            ),
        ],
    )

    measure_and_print(
        metric_name="Contextual Relevancy",
        metric=create_contextual_relevancy_metric(),
        test_case=test_case,
    )

def test_partially_useful_retrieval() -> None:
    test_case = LLMTestCase(
        input=(
            "I opened a laptop 20 days ago. Can I return it, "
            "is there a fee, and how will I receive my refund?"
        ),
        actual_output=(
            "You may return the laptop within 30 days. "
            "Your refund will be issued to the original payment method."
        ),
        expected_output=(
            "Opened laptops may be returned within 30 days. "
            "Opened laptop returns are subject to a 15% restocking fee. "
            "Approved refunds are issued to the original payment method."
        ),
        retrieval_context=[
            (
                "Northstar Electronics accepts opened laptop returns "
                "within 30 calendar days of delivery."
            ),
            (
                "Northstar Electronics was founded in 2014 and operates "
                "retail stores across several states."
            ),
            (
                "Approved refunds are issued to the customer's "
                "original payment method."
            ),
        ],
    )

    measure_and_print(
        metric_name="Contextual Precision",
        metric=create_contextual_precision_metric(),
        test_case=test_case,
    )

    # measure_and_print(
    #     metric_name="Contextual Precision",
    #     metric=create_contextual_precision_metric(),
    #     test_case=test_case,
    # )

    # measure_and_print(
    #     metric_name="Contextual Recall",
    #     metric=create_contextual_recall_metric(),
    #     test_case=test_case,
    # )


    # Leave these disabled for now.
    #
    # measure_and_print(
    #     metric_name="Contextual Precision",
    #     metric=create_contextual_precision_metric(),
    #     test_case=test_case,
    # )
    #
    # measure_and_print(
    #     metric_name="Contextual Relevancy",
    #     metric=create_contextual_relevancy_metric(),
    #     test_case=test_case,
    # )