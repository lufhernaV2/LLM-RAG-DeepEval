from evaluation.evaluation_results import CaseEvaluationResult
from evaluation.regression_analysis import (
    EvaluationComparison,
    build_regression_summary,
    calculate_score_delta,
    compare_evaluation_runs,
    create_evaluation_comparison,
    format_regression_summary,
    get_high_risk_regressions,
    get_regressions,
)
import pytest

def test_baseline_and_current_results_can_be_compared():
    baseline_result = CaseEvaluationResult(
        category="opened_laptop_return",
        risk="high",
        metric_name="FaithfulnessMetric",
        score=1.00,
        threshold=0.90,
    )

    current_result = CaseEvaluationResult(
        category="opened_laptop_return",
        risk="high",
        metric_name="FaithfulnessMetric",
        score=0.92,
        threshold=0.90,
    )

    assert baseline_result.category == current_result.category
    assert baseline_result.metric_name == current_result.metric_name

    assert baseline_result.score == 1.00
    assert current_result.score == 0.92

    assert baseline_result.passed is True
    assert current_result.passed is True


def test_score_delta_detects_quality_drop():
    baseline_result = CaseEvaluationResult(
        category="opened_laptop_return",
        risk="high",
        metric_name="FaithfulnessMetric",
        score=1.00,
        threshold=0.90,
    )

    current_result = CaseEvaluationResult(
        category="opened_laptop_return",
        risk="high",
        metric_name="FaithfulnessMetric",
        score=0.92,
        threshold=0.90,
    )

    delta = calculate_score_delta(
        baseline_result,
        current_result,
    )

    assert delta == pytest.approx(-0.08)

def test_evaluation_comparison_stores_baseline_and_current_scores():
    comparison = EvaluationComparison(
        category="opened_laptop_return",
        risk="high",
        metric_name="FaithfulnessMetric",
        baseline_score=1.00,
        current_score=0.92,
    )

    assert comparison.category == "opened_laptop_return"
    assert comparison.risk == "high"
    assert comparison.metric_name == "FaithfulnessMetric"

    assert comparison.baseline_score == 1.00
    assert comparison.current_score == 0.92

    assert comparison.delta == pytest.approx(-0.08)

def test_small_score_drop_is_tolerated():
    comparison = EvaluationComparison(
        category="opened_laptop_return",
        risk="high",
        metric_name="FaithfulnessMetric",
        baseline_score=1.00,
        current_score=0.98,
    )

    assert comparison.delta == pytest.approx(-0.02)
    assert comparison.regressed is False

def test_meaningful_score_drop_is_regression():
    comparison = EvaluationComparison(
        category="opened_laptop_return",
        risk="high",
        metric_name="FaithfulnessMetric",
        baseline_score=1.00,
        current_score=0.80,
    )

    assert comparison.delta == pytest.approx(-0.20)
    assert comparison.regressed is True

def test_case_results_convert_to_evaluation_comparison():
    baseline_result = CaseEvaluationResult(
        category="opened_laptop_return",
        risk="high",
        metric_name="FaithfulnessMetric",
        score=1.00,
        threshold=0.90,
    )

    current_result = CaseEvaluationResult(
        category="opened_laptop_return",
        risk="high",
        metric_name="FaithfulnessMetric",
        score=0.80,
        threshold=0.90,
    )

    comparison = create_evaluation_comparison(
        baseline_result,
        current_result,
    )

    assert comparison.category == "opened_laptop_return"
    assert comparison.risk == "high"
    assert comparison.metric_name == "FaithfulnessMetric"

    assert comparison.baseline_score == 1.00
    assert comparison.current_score == 0.80

    assert comparison.delta == pytest.approx(-0.20)
    assert comparison.regressed is True

def test_evaluation_runs_are_matched_by_category_and_metric():
    baseline_results = [
        CaseEvaluationResult(
            category="opened_laptop_return",
            risk="high",
            metric_name="FaithfulnessMetric",
            score=1.00,
            threshold=0.90,
        ),
        CaseEvaluationResult(
            category="refund_method",
            risk="medium",
            metric_name="AnswerRelevancyMetric",
            score=0.90,
            threshold=0.70,
        ),
    ]

    current_results = [
        CaseEvaluationResult(
            category="refund_method",
            risk="medium",
            metric_name="AnswerRelevancyMetric",
            score=0.88,
            threshold=0.70,
        ),
        CaseEvaluationResult(
            category="opened_laptop_return",
            risk="high",
            metric_name="FaithfulnessMetric",
            score=0.80,
            threshold=0.90,
        ),
    ]

    comparisons = compare_evaluation_runs(
        baseline_results,
        current_results,
    )

    opened_laptop_comparison = next(
        comparison
        for comparison in comparisons
        if comparison.category == "opened_laptop_return"
    )

    refund_method_comparison = next(
        comparison
        for comparison in comparisons
        if comparison.category == "refund_method"
    )

    assert opened_laptop_comparison.baseline_score == 1.00
    assert opened_laptop_comparison.current_score == 0.80
    assert opened_laptop_comparison.regressed is True

    assert refund_method_comparison.baseline_score == 0.90
    assert refund_method_comparison.current_score == 0.88
    assert refund_method_comparison.regressed is False

def test_regressions_can_be_filtered_by_risk():
    comparisons = [
        EvaluationComparison(
            category="opened_laptop_return",
            risk="high",
            metric_name="FaithfulnessMetric",
            baseline_score=1.00,
            current_score=0.80,
        ),
        EvaluationComparison(
            category="refund_method",
            risk="medium",
            metric_name="AnswerRelevancyMetric",
            baseline_score=0.90,
            current_score=0.82,
        ),
        EvaluationComparison(
            category="return_deadline",
            risk="high",
            metric_name="FaithfulnessMetric",
            baseline_score=0.95,
            current_score=0.93,
        ),
    ]

    regressions = get_regressions(comparisons)
    high_risk_regressions = get_high_risk_regressions(comparisons)

    assert len(regressions) == 2
    assert len(high_risk_regressions) == 1

    assert high_risk_regressions[0].category == "opened_laptop_return"

def test_regression_summary_identifies_largest_regression():
    comparisons = [
        EvaluationComparison(
            category="opened_laptop_return",
            risk="high",
            metric_name="FaithfulnessMetric",
            baseline_score=1.00,
            current_score=0.80,
        ),
        EvaluationComparison(
            category="refund_method",
            risk="medium",
            metric_name="AnswerRelevancyMetric",
            baseline_score=0.90,
            current_score=0.82,
        ),
        EvaluationComparison(
            category="return_deadline",
            risk="high",
            metric_name="FaithfulnessMetric",
            baseline_score=0.95,
            current_score=0.93,
        ),
    ]

    summary = build_regression_summary(comparisons)

    assert summary["total_comparisons"] == 3
    assert summary["regressions"] == 2
    assert summary["high_risk_regressions"] == 1

    largest_regression = summary["largest_regression"]

    assert largest_regression.category == "opened_laptop_return"
    assert largest_regression.delta == pytest.approx(-0.20)

def test_regression_summary_displays_regression_details():
    comparisons = [
        EvaluationComparison(
            category="opened_laptop_return",
            risk="high",
            metric_name="FaithfulnessMetric",
            baseline_score=1.00,
            current_score=0.80,
        ),
        EvaluationComparison(
            category="refund_method",
            risk="medium",
            metric_name="AnswerRelevancyMetric",
            baseline_score=0.90,
            current_score=0.82,
        ),
        EvaluationComparison(
            category="return_deadline",
            risk="high",
            metric_name="FaithfulnessMetric",
            baseline_score=0.95,
            current_score=0.93,
        ),
    ]

    report = format_regression_summary(comparisons)

    print()
    print(report)

    assert "Metric Comparisons: 3" in report
    assert "Regressions: 2" in report
    assert "High-Risk Regressions: 1" in report
    assert "opened_laptop_return" in report
    assert "Delta: -0.20" in report