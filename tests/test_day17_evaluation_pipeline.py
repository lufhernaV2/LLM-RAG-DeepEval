from evaluation.evaluation_results import CaseEvaluationResult
from evaluation.evaluation_pipeline import (
    get_pipeline_exit_code,
    run_evaluation_pipeline,
)
from evaluation.baseline_store import load_baseline

def test_pipeline_blocks_high_risk_regression():
    current_results = [
        CaseEvaluationResult(
            category="opened_laptop_return",
            risk="high",
            metric_name="FaithfulnessMetric",
            score=0.92,
            threshold=0.90,
        )
    ]

    pipeline_result = run_evaluation_pipeline(
        current_results=current_results,
        baseline_path="baselines/northstar_baseline.json",
    )

    assert pipeline_result.release_decision.blocked is True

    assert (
        "High-risk regression: "
        "opened_laptop_return / FaithfulnessMetric"
        in pipeline_result.release_decision.reasons
    )

    assert "Release Decision: BLOCKED" in pipeline_result.report


def test_pipeline_returns_structured_result():
    current_results = [
        CaseEvaluationResult(
            category="opened_laptop_return",
            risk="high",
            metric_name="FaithfulnessMetric",
            score=0.92,
            threshold=0.90,
        )
    ]

    pipeline_result = run_evaluation_pipeline(
        current_results=current_results,
        baseline_path="baselines/northstar_baseline.json",
    )

    assert pipeline_result.current_results == current_results
    assert len(pipeline_result.comparisons) == 1
    assert pipeline_result.release_decision.blocked is True

    assert isinstance(
        pipeline_result.report,
        str,
    )


def test_blocked_pipeline_returns_failure_exit_code():
    current_results = [
        CaseEvaluationResult(
            category="opened_laptop_return",
            risk="high",
            metric_name="FaithfulnessMetric",
            score=0.92,
            threshold=0.90,
        )
    ]

    pipeline_result = run_evaluation_pipeline(
        current_results=current_results,
        baseline_path="baselines/northstar_baseline.json",
    )

    exit_code = get_pipeline_exit_code(pipeline_result)

    assert pipeline_result.release_decision.blocked is True
    assert exit_code == 1

def test_allowed_pipeline_returns_success_exit_code():
    current_results = load_baseline(
        "baselines/northstar_baseline.json"
    )

    pipeline_result = run_evaluation_pipeline(
        current_results=current_results,
        baseline_path="baselines/northstar_baseline.json",
    )

    exit_code = get_pipeline_exit_code(pipeline_result)

    assert pipeline_result.release_decision.blocked is False
    assert exit_code == 0