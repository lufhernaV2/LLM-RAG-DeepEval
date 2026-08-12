from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric

import pytest

from evaluation.evaluation_policy import (
    build_metrics_for_risk,
    get_evaluation_thresholds,
    get_faithfulness_threshold,
)

from deepeval import assert_test

from datasets.northstar_policy_goldens import northstar_policy_goldens
from evaluation.rag_evaluation_runner import build_test_cases_from_goldens


@pytest.mark.parametrize(
    "risk, expected_threshold",
    [
        ("high", 0.90),
        ("medium", 0.70),
        ("low", 0.60),
    ],
)
def test_faithfulness_threshold_policy(risk, expected_threshold):
    threshold = get_faithfulness_threshold(risk)

    assert threshold == expected_threshold


def test_high_risk_policy_contains_all_required_metrics():
    thresholds = get_evaluation_thresholds("high")

    assert thresholds["answer_relevancy"] == 0.80
    assert thresholds["faithfulness"] == 0.90


@pytest.mark.parametrize(
    "risk, expected_answer_relevancy, expected_faithfulness",
    [
        ("high", 0.80, 0.90),
        ("medium", 0.70, 0.70),
        ("low", 0.60, 0.60),
    ],
)
def test_evaluation_policy_for_each_risk_level(
    risk,
    expected_answer_relevancy,
    expected_faithfulness,
):
    thresholds = get_evaluation_thresholds(risk)

    assert thresholds["answer_relevancy"] == expected_answer_relevancy
    assert thresholds["faithfulness"] == expected_faithfulness


def test_build_metrics_for_high_risk_policy():
    metrics = build_metrics_for_risk("high")

    answer_relevancy_metric = next(
        metric
        for metric in metrics
        if isinstance(metric, AnswerRelevancyMetric)
    )

    faithfulness_metric = next(
        metric
        for metric in metrics
        if isinstance(metric, FaithfulnessMetric)
    )

    assert answer_relevancy_metric.threshold == 0.80
    assert faithfulness_metric.threshold == 0.90


@pytest.mark.parametrize(
    "golden, test_case",
    list(
        zip(
            northstar_policy_goldens,
            build_test_cases_from_goldens(northstar_policy_goldens),
        )
    ),
)
def test_dataset_uses_centralized_evaluation_policy(golden, test_case):
    risk = golden.additional_metadata["risk"]

    metrics = build_metrics_for_risk(risk)

    assert_test(
        test_case=test_case,
        metrics=metrics,
    )