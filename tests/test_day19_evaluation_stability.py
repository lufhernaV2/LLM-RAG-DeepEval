from evaluation.evaluation_results import CaseEvaluationResult
from evaluation.evaluation_stability import (
    BORDERLINE,
    CLEAR_FAILURE,
    CLEAR_PASS,
    CONFIRMED_FAILURE,
    CONFIRMED_PASS,
    UNSTABLE,
    create_stability_result,
    classify_result_stability,
    get_borderline_results,
    build_confirmation_requests,
    run_confirmation,
    run_targeted_confirmations,
    EvaluationStabilityResult,
    should_block_for_stability,
    ConfirmationRequest
)
from datasets.northstar_policy_goldens import northstar_policy_goldens
from evaluation.evaluation_policy import build_metrics_for_risk
from evaluation.rag_evaluation_runner import build_test_cases_from_goldens
from evaluation.release_decision import build_release_decision
from evaluation.baseline_store import load_baseline
from evaluation.evaluation_pipeline import run_evaluation_pipeline


def test_stability_policy_blocks_failure_and_unstable_results():
    confirmed_pass = EvaluationStabilityResult(
        category="opened_laptop_return",
        risk="high",
        metric_name="FaithfulnessMetric",
        initial_score=0.91,
        confirmation_score=0.95,
        threshold=0.90,
    )

    assert should_block_for_stability(
        [confirmed_pass]
    ) is False

    confirmed_failure = EvaluationStabilityResult(
        category="opened_laptop_return",
        risk="high",
        metric_name="FaithfulnessMetric",
        initial_score=0.89,
        confirmation_score=0.88,
        threshold=0.90,
    )

    assert should_block_for_stability(
        [confirmed_failure]
    ) is True

    unstable = EvaluationStabilityResult(
        category="opened_laptop_return",
        risk="high",
        metric_name="FaithfulnessMetric",
        initial_score=0.89,
        confirmation_score=1.00,
        threshold=0.90,
    )

    assert should_block_for_stability(
        [unstable]
    ) is True

def test_runs_only_requested_confirmation_metric():
    class FakeAnswerRelevancyMetric:
        def __init__(self):
            self.score = None
            self.threshold = 0.80

        def measure(self, test_case):
            raise AssertionError(
                "Answer Relevancy should not be rerun"
            )

    class FaithfulnessMetric:
        def __init__(self):
            self.score = None
            self.threshold = 0.90

        def measure(self, test_case):
            self.score = 0.88

    def fake_build_metrics_for_risk(risk):
        assert risk == "high"

        return [
            FakeAnswerRelevancyMetric(),
            FaithfulnessMetric(),
        ]

    request = ConfirmationRequest(
        category="opened_laptop_return",
        risk="high",
        metric_name="FaithfulnessMetric",
        initial_score=0.89,
        threshold=0.90,
    )

    fake_test_case = object()

    stability_result = run_confirmation(
        confirmation_request=request,
        test_case=fake_test_case,
        build_metrics_for_risk=fake_build_metrics_for_risk,
    )

    assert stability_result.initial_score == 0.89
    assert stability_result.confirmation_score == 0.88
    assert stability_result.outcome == CONFIRMED_FAILURE

def test_confirmation_results_distinguish_stable_and_unstable_outcomes():
    passing_request = ConfirmationRequest(
        category="opened_laptop_return",
        risk="high",
        metric_name="FaithfulnessMetric",
        initial_score=0.91,
        threshold=0.90,
    )

    confirmed_pass = create_stability_result(
        confirmation_request=passing_request,
        confirmation_score=0.92,
    )

    assert confirmed_pass.outcome == CONFIRMED_PASS

    failing_request = ConfirmationRequest(
        category="opened_laptop_return",
        risk="high",
        metric_name="FaithfulnessMetric",
        initial_score=0.89,
        threshold=0.90,
    )

    confirmed_failure = create_stability_result(
        confirmation_request=failing_request,
        confirmation_score=0.88,
    )

    assert confirmed_failure.outcome == CONFIRMED_FAILURE

    unstable_result = create_stability_result(
        confirmation_request=failing_request,
        confirmation_score=0.91,
    )

    assert unstable_result.outcome == UNSTABLE

def test_classifies_clear_pass_borderline_and_clear_failure():
    threshold = 0.90

    assert classify_result_stability(
        score=0.96,
        threshold=threshold,
    ) == CLEAR_PASS

    assert classify_result_stability(
        score=0.91,
        threshold=threshold,
    ) == BORDERLINE

    assert classify_result_stability(
        score=0.89,
        threshold=threshold,
    ) == BORDERLINE

    assert classify_result_stability(
        score=0.70,
        threshold=threshold,
    ) == CLEAR_FAILURE


def test_gets_only_borderline_results():
    results = [
        CaseEvaluationResult(
            category="opened_laptop_return",
            risk="high",
            metric_name="AnswerRelevancyMetric",
            score=0.96,
            threshold=0.80,
        ),
        CaseEvaluationResult(
            category="opened_laptop_return",
            risk="high",
            metric_name="FaithfulnessMetric",
            score=0.89,
            threshold=0.90,
        ),
        CaseEvaluationResult(
            category="refund_method",
            risk="medium",
            metric_name="FaithfulnessMetric",
            score=0.40,
            threshold=0.70,
        ),
    ]

    borderline_results = get_borderline_results(results)

    assert len(borderline_results) == 1

    borderline_result = borderline_results[0]

    assert borderline_result.category == "opened_laptop_return"
    assert borderline_result.metric_name == "FaithfulnessMetric"
    assert borderline_result.score == 0.89
    assert borderline_result.threshold == 0.90

def test_builds_confirmation_request_only_for_borderline_result():
    results = [
        CaseEvaluationResult(
            category="opened_laptop_return",
            risk="high",
            metric_name="AnswerRelevancyMetric",
            score=0.98,
            threshold=0.80,
        ),
        CaseEvaluationResult(
            category="opened_laptop_return",
            risk="high",
            metric_name="FaithfulnessMetric",
            score=0.89,
            threshold=0.90,
        ),
        CaseEvaluationResult(
            category="refund_method",
            risk="medium",
            metric_name="FaithfulnessMetric",
            score=0.40,
            threshold=0.70,
        ),
    ]

    confirmation_requests = build_confirmation_requests(results)

    assert len(confirmation_requests) == 1

    request = confirmation_requests[0]

    assert request.category == "opened_laptop_return"
    assert request.risk == "high"
    assert request.metric_name == "FaithfulnessMetric"
    assert request.initial_score == 0.89
    assert request.threshold == 0.90

def test_real_northstar_faithfulness_confirmation():
    test_cases = build_test_cases_from_goldens(
        northstar_policy_goldens
    )

    opened_laptop_test_case = None

    for test_case in test_cases:
        if (
            test_case.additional_metadata["category"]
            == "opened_laptop_return"
        ):
            opened_laptop_test_case = test_case
            break

    assert opened_laptop_test_case is not None

    request = ConfirmationRequest(
        category="opened_laptop_return",
        risk="high",
        metric_name="FaithfulnessMetric",
        initial_score=0.89,
        threshold=0.90,
    )

    stability_result = run_confirmation(
        confirmation_request=request,
        test_case=opened_laptop_test_case,
        build_metrics_for_risk=build_metrics_for_risk,
    )

    print()
    print("=== Live Confirmation Result ===")
    print(
        f"Category: {stability_result.category}"
    )
    print(
        f"Metric: {stability_result.metric_name}"
    )
    print(
        f"Initial Score: {stability_result.initial_score:.2f}"
    )
    print(
        f"Confirmation Score: "
        f"{stability_result.confirmation_score:.2f}"
    )
    print(
        f"Threshold: {stability_result.threshold:.2f}"
    )
    print(
        f"Outcome: {stability_result.outcome}"
    )

    assert stability_result.category == "opened_laptop_return"
    assert stability_result.metric_name == "FaithfulnessMetric"

    assert 0.0 <= stability_result.confirmation_score <= 1.0

    assert stability_result.outcome in {
        CONFIRMED_PASS,
        CONFIRMED_FAILURE,
        UNSTABLE,
    }

def test_unstable_evaluation_blocks_release_decision():
    current_results = [
        CaseEvaluationResult(
            category="opened_laptop_return",
            risk="high",
            metric_name="FaithfulnessMetric",
            score=0.91,
            threshold=0.90,
        )
    ]

    stability_result = EvaluationStabilityResult(
        category="opened_laptop_return",
        risk="high",
        metric_name="FaithfulnessMetric",
        initial_score=0.89,
        confirmation_score=0.91,
        threshold=0.90,
    )

    decision = build_release_decision(
        current_results=current_results,
        comparisons=[],
        stability_results=[stability_result],
    )

    assert current_results[0].passed is True
    assert stability_result.outcome == UNSTABLE

    assert decision.blocked is True

    assert decision.reasons == [
        "Unstable evaluation: "
        "opened_laptop_return / FaithfulnessMetric"
    ]

def test_pipeline_blocks_when_stability_is_unstable():
    current_results = load_baseline(
        "baselines/northstar_baseline.json"
    )

    stability_result = EvaluationStabilityResult(
        category="opened_laptop_return",
        risk="high",
        metric_name="FaithfulnessMetric",
        initial_score=0.89,
        confirmation_score=1.00,
        threshold=0.90,
    )

    pipeline_result = run_evaluation_pipeline(
        current_results=current_results,
        baseline_path="baselines/northstar_baseline.json",
        stability_results=[stability_result],
    )

    assert stability_result.outcome == UNSTABLE

    assert pipeline_result.release_decision.blocked is True

    assert (
        "Unstable evaluation: "
        "opened_laptop_return / FaithfulnessMetric"
        in pipeline_result.release_decision.reasons
    )

    assert "Release Decision: BLOCKED" in pipeline_result.report

def test_targeted_confirmations_only_run_for_borderline_results():
    class FakeTestCase:
        def __init__(self, category):
            self.additional_metadata = {
                "category": category
            }

    class AnswerRelevancyMetric:
        def __init__(self):
            self.score = None
            self.threshold = 0.80

        def measure(self, test_case):
            raise AssertionError(
                "Answer Relevancy should not be confirmed"
            )

    class FaithfulnessMetric:
        def __init__(self):
            self.score = None
            self.threshold = 0.90

        def measure(self, test_case):
            self.score = 0.88

    def fake_build_metrics_for_risk(risk):
        assert risk == "high"

        return [
            AnswerRelevancyMetric(),
            FaithfulnessMetric(),
        ]

    current_results = [
        CaseEvaluationResult(
            category="opened_laptop_return",
            risk="high",
            metric_name="AnswerRelevancyMetric",
            score=0.98,
            threshold=0.80,
        ),
        CaseEvaluationResult(
            category="opened_laptop_return",
            risk="high",
            metric_name="FaithfulnessMetric",
            score=0.89,
            threshold=0.90,
        ),
    ]

    test_cases = [
        FakeTestCase(
            category="opened_laptop_return"
        )
    ]

    stability_results = run_targeted_confirmations(
        current_results=current_results,
        test_cases=test_cases,
        build_metrics_for_risk=fake_build_metrics_for_risk,
    )

    assert len(stability_results) == 1

    result = stability_results[0]

    assert result.metric_name == "FaithfulnessMetric"
    assert result.initial_score == 0.89
    assert result.confirmation_score == 0.88
    assert result.outcome == CONFIRMED_FAILURE