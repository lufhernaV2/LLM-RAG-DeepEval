from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from evaluation.evaluation_results import (
    CaseEvaluationResult,
    build_evaluation_summary,
    create_case_evaluation_result,
    create_results_for_metrics,
    evaluate_dataset_to_results,
    format_evaluation_summary,
    get_failed_results,
    get_high_risk_failures,
    should_block_release,
)

from evaluation.evaluation_policy import build_metrics_for_risk
from datasets.northstar_policy_goldens import northstar_policy_goldens
from evaluation.rag_evaluation_runner import build_test_cases_from_goldens

def test_case_evaluation_result_stores_evaluation_data():
    result = CaseEvaluationResult(
        category="opened_laptop_return",
        risk="high",
        metric_name="Faithfulness",
        score=0.50,
        threshold=0.90,
    )

    assert result.category == "opened_laptop_return"
    assert result.risk == "high"
    assert result.metric_name == "Faithfulness"
    assert result.score == 0.50
    assert result.threshold == 0.90
    assert result.passed is False

def test_case_evaluation_result_passes_when_score_meets_threshold():
    result = CaseEvaluationResult(
        category="opened_laptop_return",
        risk="high",
        metric_name="Faithfulness",
        score=1.00,
        threshold=0.90,
    )

    assert result.passed is True

def test_measured_metric_converts_to_structured_result():
    test_case = LLMTestCase(
        input="Can I return my opened laptop?",
        actual_output=(
            "Yes. Opened laptops may be returned within 30 days, "
            "but a 15% restocking fee applies."
        ),
        retrieval_context=[
            "Opened laptops may be returned within 30 days.",
            "A 15% restocking fee applies to opened laptops.",
        ],
    )

    metric = FaithfulnessMetric(
        threshold=0.90,
        model="gpt-4.1-mini",
        include_reason=True,
    )

    metric.measure(test_case)

    result = create_case_evaluation_result(
        category="opened_laptop_return",
        risk="high",
        metric=metric,
    )

    assert result.category == "opened_laptop_return"
    assert result.risk == "high"
    assert result.metric_name == "FaithfulnessMetric"
    assert result.score == metric.score
    assert result.threshold == 0.90
    assert result.passed is True

def test_one_rag_case_creates_multiple_structured_results():
    test_case = LLMTestCase(
        input="Can I return my opened laptop?",
        actual_output=(
            "Yes. Opened laptops may be returned within 30 days, "
            "but a 15% restocking fee applies."
        ),
        retrieval_context=[
            "Opened laptops may be returned within 30 days.",
            "A 15% restocking fee applies to opened laptops.",
        ],
    )

    metrics = [
        AnswerRelevancyMetric(
            threshold=0.80,
            model="gpt-4.1-mini",
            include_reason=True,
        ),
        FaithfulnessMetric(
            threshold=0.90,
            model="gpt-4.1-mini",
            include_reason=True,
        ),
    ]

    results = create_results_for_metrics(
        category="opened_laptop_return",
        risk="high",
        test_case=test_case,
        metrics=metrics,
    )

    assert len(results) == 2

    assert results[0].metric_name == "AnswerRelevancyMetric"
    assert results[1].metric_name == "FaithfulnessMetric"

    assert results[0].threshold == 0.80
    assert results[1].threshold == 0.90

    assert results[0].passed is True
    assert results[1].passed is True

def test_structured_results_use_centralized_policy():
    test_case = LLMTestCase(
        input="Can I return my opened laptop?",
        actual_output=(
            "Yes. Opened laptops may be returned within 30 days, "
            "but a 15% restocking fee applies."
        ),
        retrieval_context=[
            "Opened laptops may be returned within 30 days.",
            "A 15% restocking fee applies to opened laptops.",
        ],
    )

    risk = "high"

    metrics = build_metrics_for_risk(risk)

    results = create_results_for_metrics(
        category="opened_laptop_return",
        risk=risk,
        test_case=test_case,
        metrics=metrics,
    )

    assert len(results) == 2

    assert results[0].threshold == 0.80
    assert results[1].threshold == 0.90

    assert results[0].passed is True
    assert results[1].passed is True

def test_dataset_creates_structured_evaluation_results():
    test_cases = build_test_cases_from_goldens(
        northstar_policy_goldens
    )

    results = evaluate_dataset_to_results(
        goldens=northstar_policy_goldens,
        test_cases=test_cases,
        build_metrics_for_risk=build_metrics_for_risk,
    )

    assert len(results) == 6

    assert all(
        isinstance(result, CaseEvaluationResult)
        for result in results
    )

    assert all(
        result.category
        for result in results
    )

    assert all(
        result.risk
        for result in results
    )

def test_failed_results_can_be_filtered_by_risk():
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
            score=0.60,
            threshold=0.70,
        ),
        CaseEvaluationResult(
            category="return_deadline",
            risk="high",
            metric_name="FaithfulnessMetric",
            score=1.00,
            threshold=0.90,
        ),
    ]

    failed_results = get_failed_results(results)
    high_risk_failures = get_high_risk_failures(results)

    assert len(failed_results) == 2
    assert len(high_risk_failures) == 1

    assert high_risk_failures[0].category == "opened_laptop_return"


def test_release_is_blocked_when_high_risk_failure_exists():
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
            score=1.00,
            threshold=0.70,
        ),
    ]

    assert should_block_release(results) is True


def test_release_is_allowed_when_no_high_risk_failures_exist():
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
            score=0.60,
            threshold=0.70,
        ),
    ]

    assert should_block_release(results) is False


def test_evaluation_summary_reports_release_status():
    results = [
        CaseEvaluationResult(
            category="opened_laptop_return",
            risk="high",
            metric_name="AnswerRelevancyMetric",
            score=1.00,
            threshold=0.80,
        ),
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
            score=1.00,
            threshold=0.70,
        ),
    ]

    summary = build_evaluation_summary(results)

    assert summary["total_results"] == 3
    assert summary["passed"] == 2
    assert summary["failed"] == 1
    assert summary["high_risk_failures"] == 1
    assert summary["release_decision"] == "BLOCKED"

def test_release_summary_displays_failure_details():
    results = [
        CaseEvaluationResult(
            category="opened_laptop_return",
            risk="high",
            metric_name="AnswerRelevancyMetric",
            score=1.00,
            threshold=0.80,
        ),
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
            score=1.00,
            threshold=0.70,
        ),
    ]

    report = format_evaluation_summary(results)

    print()
    print(report)

    assert "Release Decision: BLOCKED" in report
    assert "opened_laptop_return" in report
    assert "FaithfulnessMetric" in report