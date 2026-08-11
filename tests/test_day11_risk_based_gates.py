import pytest

from datasets.northstar_policy_goldens import northstar_policy_goldens
from deepeval import assert_test
from deepeval.metrics import FaithfulnessMetric

from evaluation.rag_evaluation_runner import build_test_cases_from_goldens


@pytest.mark.parametrize("golden", northstar_policy_goldens)
def test_goldens_have_valid_risk_metadata(golden):
    risk = golden.additional_metadata["risk"]

    assert risk in {"high", "medium", "low"}

def get_faithfulness_threshold(risk):
    thresholds = {
        "high": 0.90,
        "medium": 0.70,
        "low": 0.60,
    }

    return thresholds[risk]

@pytest.mark.parametrize(
    "risk, expected_threshold",
    [
        ("high", 0.90),
        ("medium", 0.70),
        ("low", 0.60),
    ],
)
def test_risk_maps_to_expected_faithfulness_threshold(
    risk,
    expected_threshold,
):
    threshold = get_faithfulness_threshold(risk)

    assert threshold == expected_threshold


@pytest.mark.parametrize("golden", northstar_policy_goldens)
def test_each_golden_gets_a_faithfulness_threshold(golden):
    risk = golden.additional_metadata["risk"]

    threshold = get_faithfulness_threshold(risk)

    assert threshold > 0
    assert threshold <= 1


@pytest.mark.parametrize(
    "golden, test_case",
    list(
        zip(
            northstar_policy_goldens,
            build_test_cases_from_goldens(northstar_policy_goldens),
        )
    ),
)
def test_goldens_pass_risk_based_faithfulness_gate(golden, test_case):
    risk = golden.additional_metadata["risk"]

    threshold = get_faithfulness_threshold(risk)

    faithfulness_metric = FaithfulnessMetric(
        threshold=threshold,
        model="gpt-4.1-mini",
        include_reason=True,
    )

    assert_test(
        test_case=test_case,
        metrics=[faithfulness_metric],
    )

def passes_faithfulness_gate(score, risk):
    threshold = get_faithfulness_threshold(risk)

    return score >= threshold

def test_same_score_has_different_result_based_on_risk():
    score = 0.80

    medium_risk_result = passes_faithfulness_gate(
        score=score,
        risk="medium",
    )

    high_risk_result = passes_faithfulness_gate(
        score=score,
        risk="high",
    )

    assert medium_risk_result is True
    assert high_risk_result is False


def test_high_risk_gate_detects_regression():
    test_cases = build_test_cases_from_goldens(
        northstar_policy_goldens
    )

    golden_test_case_pairs = list(
        zip(
            northstar_policy_goldens,
            test_cases,
        )
    )

    golden, test_case = next(
        (golden, test_case)
        for golden, test_case in golden_test_case_pairs
        if golden.additional_metadata["category"] == "opened_laptop_return"
    )

    risk = golden.additional_metadata["risk"]

    threshold = get_faithfulness_threshold(risk)

    test_case.actual_output = (
        "Yes, you can return the opened laptop within 30 days, "
        "and there is no restocking fee."
    )

    faithfulness_metric = FaithfulnessMetric(
        threshold=threshold,
        model="gpt-4.1-mini",
        include_reason=True,
    )

    faithfulness_metric.measure(test_case)

    assert risk == "high"
    assert threshold == 0.90
    assert faithfulness_metric.score < threshold