from pathlib import Path

from datasets.northstar_policy_goldens import northstar_policy_goldens
from evaluation.evaluation_policy import build_metrics_for_risk
from evaluation.evaluation_results import evaluate_dataset_to_results
from evaluation.rag_evaluation_runner import build_test_cases_from_goldens
from evaluation.baseline_store import (
    can_promote_to_baseline,
    dict_to_result,
    load_baseline,
    result_to_dict,
    save_baseline,
)
from evaluation.evaluation_results import CaseEvaluationResult
from copy import deepcopy

from evaluation.regression_analysis import (
    compare_evaluation_runs,
    format_regression_summary,
)


def test_case_evaluation_result_serializes_to_dictionary():
    result = CaseEvaluationResult(
        category="opened_laptop_return",
        risk="high",
        metric_name="FaithfulnessMetric",
        score=1.00,
        threshold=0.90,
    )

    serialized = result_to_dict(result)

    assert serialized == {
        "category": "opened_laptop_return",
        "risk": "high",
        "metric_name": "FaithfulnessMetric",
        "score": 1.00,
        "threshold": 0.90,
    }

def test_dictionary_deserializes_to_case_evaluation_result():
    data = {
        "category": "opened_laptop_return",
        "risk": "high",
        "metric_name": "FaithfulnessMetric",
        "score": 1.00,
        "threshold": 0.90,
    }

    result = dict_to_result(data)

    assert isinstance(result, CaseEvaluationResult)
    assert result.category == "opened_laptop_return"
    assert result.risk == "high"
    assert result.metric_name == "FaithfulnessMetric"
    assert result.score == 1.00
    assert result.threshold == 0.90
    assert result.passed is True


def test_evaluation_result_round_trip_preserves_data():
    original_result = CaseEvaluationResult(
        category="opened_laptop_return",
        risk="high",
        metric_name="FaithfulnessMetric",
        score=0.95,
        threshold=0.90,
    )

    serialized = result_to_dict(original_result)
    restored_result = dict_to_result(serialized)

    assert restored_result.category == original_result.category
    assert restored_result.risk == original_result.risk
    assert restored_result.metric_name == original_result.metric_name
    assert restored_result.score == original_result.score
    assert restored_result.threshold == original_result.threshold
    assert restored_result.passed == original_result.passed

def test_baseline_results_can_be_saved_to_json(tmp_path):
    results = [
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
            score=0.95,
            threshold=0.70,
        ),
    ]

    baseline_file = tmp_path / "baseline.json"

    save_baseline(
        results=results,
        file_path=baseline_file,
    )

    assert baseline_file.exists()

    contents = baseline_file.read_text(
        encoding="utf-8"
    )

    assert "opened_laptop_return" in contents
    assert "FaithfulnessMetric" in contents
    assert "refund_method" in contents

def test_saved_baseline_can_be_loaded_as_results(tmp_path):
    original_results = [
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
            score=0.95,
            threshold=0.70,
        ),
    ]

    baseline_file = tmp_path / "baseline.json"

    save_baseline(
        results=original_results,
        file_path=baseline_file,
    )

    loaded_results = load_baseline(baseline_file)

    assert len(loaded_results) == 2

    assert loaded_results[0].category == "opened_laptop_return"
    assert loaded_results[0].metric_name == "FaithfulnessMetric"
    assert loaded_results[0].score == 1.00
    assert loaded_results[0].threshold == 0.90
    assert loaded_results[0].passed is True

    assert loaded_results[1].category == "refund_method"
    assert loaded_results[1].metric_name == "AnswerRelevancyMetric"
    assert loaded_results[1].score == 0.95

def test_persisted_baseline_can_detect_current_regression(tmp_path):
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
            score=0.95,
            threshold=0.70,
        ),
    ]

    baseline_file = tmp_path / "baseline.json"

    save_baseline(
        results=baseline_results,
        file_path=baseline_file,
    )

    loaded_baseline = load_baseline(baseline_file)

    current_results = [
        CaseEvaluationResult(
            category="opened_laptop_return",
            risk="high",
            metric_name="FaithfulnessMetric",
            score=0.80,
            threshold=0.90,
        ),
        CaseEvaluationResult(
            category="refund_method",
            risk="medium",
            metric_name="AnswerRelevancyMetric",
            score=0.93,
            threshold=0.70,
        ),
    ]

    comparisons = compare_evaluation_runs(
        loaded_baseline,
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

    assert refund_method_comparison.baseline_score == 0.95
    assert refund_method_comparison.current_score == 0.93
    assert refund_method_comparison.regressed is False

def test_high_risk_failure_cannot_be_promoted_to_baseline():
    results = [
        CaseEvaluationResult(
            category="opened_laptop_return",
            risk="high",
            metric_name="FaithfulnessMetric",
            score=0.50,
            threshold=0.90,
        ),
        CaseEvaluationResult(
            category="refund_method",
            risk="medium",
            metric_name="AnswerRelevancyMetric",
            score=0.95,
            threshold=0.70,
        ),
    ]

    assert can_promote_to_baseline(results) is False

def test_healthy_run_can_be_promoted_to_baseline():
    results = [
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
            score=0.95,
            threshold=0.70,
        ),
    ]

    assert can_promote_to_baseline(results) is True

def test_real_northstar_evaluation_can_be_saved_as_baseline():
    test_cases = build_test_cases_from_goldens(
        northstar_policy_goldens
    )

    results = evaluate_dataset_to_results(
        goldens=northstar_policy_goldens,
        test_cases=test_cases,
        build_metrics_for_risk=build_metrics_for_risk,
    )

    assert len(results) > 0

    assert can_promote_to_baseline(results) is True

    baseline_file = Path(
        "baselines/northstar_baseline.json"
    )

    save_baseline(
        results=results,
        file_path=baseline_file,
    )

    assert baseline_file.exists()

    loaded_baseline = load_baseline(baseline_file)

    assert len(loaded_baseline) == len(results)

    print()
    print(
        f"Saved {len(loaded_baseline)} metric results "
        f"to {baseline_file}"
    )

def test_saved_northstar_baseline_detects_future_regression():
    baseline_file = Path(
        "baselines/northstar_baseline.json"
    )

    baseline_results = load_baseline(baseline_file)

    current_results = deepcopy(baseline_results)

    target_result = next(
        result
        for result in current_results
        if result.category == "opened_laptop_return"
        and result.metric_name == "FaithfulnessMetric"
    )

    target_result.score = max(
        0.0,
        target_result.score - 0.20,
    )

    comparisons = compare_evaluation_runs(
        baseline_results,
        current_results,
    )

    report = format_regression_summary(comparisons)

    print()
    print(report)

    opened_laptop_comparison = next(
        comparison
        for comparison in comparisons
        if comparison.category == "opened_laptop_return"
        and comparison.metric_name == "FaithfulnessMetric"
    )

    assert opened_laptop_comparison.regressed is True