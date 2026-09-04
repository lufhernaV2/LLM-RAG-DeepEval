import pytest
from evaluation.baseline_store import load_baseline

from evaluation.evaluation_results import (
    CaseEvaluationResult,
    should_block_release,
)
from evaluation.regression_analysis import (
    compare_evaluation_runs,
    get_high_risk_regressions,
)

from evaluation.release_decision import (
    build_release_decision,
    format_release_report,
    should_block_release_with_regressions,
)


def test_high_risk_regression_can_exist_even_when_quality_gate_passes():
    baseline_results = [
        CaseEvaluationResult(
            category="opened_laptop_return",
            risk="high",
            metric_name="FaithfulnessMetric",
            score=1.00,
            threshold=0.90,
        )
    ]

    current_results = [
        CaseEvaluationResult(
            category="opened_laptop_return",
            risk="high",
            metric_name="FaithfulnessMetric",
            score=0.92,
            threshold=0.90,
        )
    ]

    # The current result still satisfies the absolute quality threshold.
    assert current_results[0].passed is True

    # Therefore the existing threshold-based release gate allows it.
    assert should_block_release(current_results) is False

    comparisons = compare_evaluation_runs(
        baseline_results,
        current_results,
    )

    high_risk_regressions = get_high_risk_regressions(comparisons)

    assert len(high_risk_regressions) == 1

    regression = high_risk_regressions[0]

    assert regression.category == "opened_laptop_return"
    assert regression.metric_name == "FaithfulnessMetric"
    assert regression.baseline_score == 1.00
    assert regression.current_score == 0.92
    assert regression.delta == pytest.approx(-0.08)


def test_high_risk_regression_blocks_release_even_when_threshold_passes():
    baseline_results = [
        CaseEvaluationResult(
            category="opened_laptop_return",
            risk="high",
            metric_name="FaithfulnessMetric",
            score=1.00,
            threshold=0.90,
        )
    ]

    current_results = [
        CaseEvaluationResult(
            category="opened_laptop_return",
            risk="high",
            metric_name="FaithfulnessMetric",
            score=0.92,
            threshold=0.90,
        )
    ]

    comparisons = compare_evaluation_runs(
        baseline_results,
        current_results,
    )

    assert current_results[0].passed is True
    assert should_block_release(current_results) is False

    assert should_block_release_with_regressions(
        current_results,
        comparisons,
    ) is True

def test_release_decision_explains_high_risk_regression():
    baseline_results = [
        CaseEvaluationResult(
            category="opened_laptop_return",
            risk="high",
            metric_name="FaithfulnessMetric",
            score=1.00,
            threshold=0.90,
        )
    ]

    current_results = [
        CaseEvaluationResult(
            category="opened_laptop_return",
            risk="high",
            metric_name="FaithfulnessMetric",
            score=0.92,
            threshold=0.90,
        )
    ]

    comparisons = compare_evaluation_runs(
        baseline_results,
        current_results,
    )

    decision = build_release_decision(
        current_results,
        comparisons,
    )

    assert decision.blocked is True
    assert decision.reasons == [
        "High-risk regression: opened_laptop_return / FaithfulnessMetric"
    ]

def test_real_northstar_baseline_can_drive_release_decision():
    baseline_results = load_baseline(
        "baselines/northstar_baseline.json"
    )

    current_results = []

    for result in baseline_results:
        score = result.score

        if (
            result.category == "opened_laptop_return"
            and result.metric_name == "FaithfulnessMetric"
        ):
            score = 0.80

        current_results.append(
            CaseEvaluationResult(
                category=result.category,
                risk=result.risk,
                metric_name=result.metric_name,
                score=score,
                threshold=result.threshold,
            )
        )

    comparisons = compare_evaluation_runs(
        baseline_results,
        current_results,
    )

    decision = build_release_decision(
        current_results,
        comparisons,
    )

    assert len(baseline_results) == 6
    assert len(current_results) == 6

    assert decision.blocked is True

    assert (
        "High-risk regression: "
        "opened_laptop_return / FaithfulnessMetric"
        in decision.reasons
    )

def test_real_northstar_release_report():
    baseline_results = load_baseline(
        "baselines/northstar_baseline.json"
    )

    current_results = []

    for result in baseline_results:
        score = result.score

        if (
            result.category == "opened_laptop_return"
            and result.metric_name == "FaithfulnessMetric"
        ):
            score = 0.80

        current_results.append(
            CaseEvaluationResult(
                category=result.category,
                risk=result.risk,
                metric_name=result.metric_name,
                score=score,
                threshold=result.threshold,
            )
        )

    comparisons = compare_evaluation_runs(
        baseline_results,
        current_results,
    )

    report = format_release_report(
        current_results,
        comparisons,
    )

    print()
    print(report)

    assert "Metric Evaluations: 6" in report
    assert "Threshold Failures: 1" in report
    assert "High-Risk Threshold Failures: 1" in report
    assert "Regressions: 1" in report
    assert "High-Risk Regressions: 1" in report
    assert "Release Decision: BLOCKED" in report

    assert (
        "High-risk threshold failure: "
        "opened_laptop_return / FaithfulnessMetric"
        in report
    )

    assert (
        "High-risk regression: "
        "opened_laptop_return / FaithfulnessMetric"
        in report
    )