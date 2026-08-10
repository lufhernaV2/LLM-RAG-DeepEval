from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
import pytest
from deepeval.test_case import LLMTestCase
from datasets.northstar_policy_goldens import northstar_policy_goldens
from evaluation.rag_evaluation_runner import build_test_cases_from_goldens

# @pytest.mark.xfail(
#     reason="Controlled bad response used to verify the faithfulness quality gate"
# )
# def test_critical_policy_response_is_faithful():
#     test_case = LLMTestCase(
#         input="I opened a laptop 20 days ago. Can I return it?",
#         actual_output=(
#             "Yes, you can return the laptop within 30 days. "
#             "There is no restocking fee."
#         ),
#         retrieval_context=[
#             "Opened laptops may be returned within 30 days.",
#             "A 15% restocking fee applies to opened laptops.",
#         ],
#     )

#     faithfulness_metric = FaithfulnessMetric(
#         threshold=0.7,
#         model="gpt-4.1-mini",
#         include_reason=True,
#     )

#     assert_test(
#         test_case=test_case,
#         metrics=[faithfulness_metric],
#     )


def test_critical_policy_response_passes_when_faithful():
    test_case = LLMTestCase(
        input="I opened a laptop 20 days ago. Can I return it?",
        actual_output=(
            "Yes. Opened laptops may be returned within 30 days, "
            "but a 15% restocking fee applies."
        ),
        retrieval_context=[
            "Opened laptops may be returned within 30 days.",
            "A 15% restocking fee applies to opened laptops.",
        ],
    )

    faithfulness_metric = FaithfulnessMetric(
        threshold=0.7,
        model="gpt-4.1-mini",
        include_reason=True,
    )

    assert_test(
        test_case=test_case,
        metrics=[faithfulness_metric],
    )

@pytest.mark.parametrize(
    "test_case",
    build_test_cases_from_goldens(northstar_policy_goldens),
)
def test_northstar_policy_dataset_passes_quality_gate(test_case):

    faithfulness_metric = FaithfulnessMetric(
        threshold=0.7,
        model="gpt-4.1-mini",
        include_reason=True,
    )

    answer_relevancy_metric = AnswerRelevancyMetric(
        threshold=0.7,
        model="gpt-4.1-mini",
        include_reason=True,
    )

    assert_test(
        test_case=test_case,
        metrics=[
            answer_relevancy_metric,
            faithfulness_metric,
        ],
    )


def test_faithfulness_detects_critical_regression():
    test_cases = build_test_cases_from_goldens(northstar_policy_goldens)

    regression_case = test_cases[0]

    regression_case.actual_output = (
        "Yes, you can return the opened laptop within 30 days, "
        "and there is no restocking fee."
    )

    faithfulness_metric = FaithfulnessMetric(
        threshold=0.7,
        model="gpt-4.1-mini",
        include_reason=True,
    )

    faithfulness_metric.measure(regression_case)

    assert faithfulness_metric.score < faithfulness_metric.threshold