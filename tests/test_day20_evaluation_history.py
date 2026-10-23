from evaluation.evaluation_history import (
    append_evaluation_run,
    create_evaluation_run_record,
    load_evaluation_history,
    get_metric_history,
)
from evaluation.evaluation_results import CaseEvaluationResult
from evaluation.release_decision import ReleaseDecision


def test_creates_evaluation_run_record():
    results = [
        CaseEvaluationResult(
            category="opened_laptop_return",
            risk="high",
            metric_name="FaithfulnessMetric",
            score=1.00,
            threshold=0.90,
        )
    ]

    release_decision = ReleaseDecision(
        blocked=False,
        reasons=[],
    )

    run_record = create_evaluation_run_record(
        run_id="run-001",
        results=results,
        release_decision=release_decision,
    )

    assert run_record.run_id == "run-001"
    assert run_record.results == results
    assert run_record.release_blocked is False
    assert run_record.timestamp

def test_appends_multiple_evaluation_runs(tmp_path):
    history_path = tmp_path / "northstar_history.json"

    results = [
        CaseEvaluationResult(
            category="opened_laptop_return",
            risk="high",
            metric_name="FaithfulnessMetric",
            score=1.00,
            threshold=0.90,
        )
    ]

    allowed_decision = ReleaseDecision(
        blocked=False,
        reasons=[],
    )

    blocked_decision = ReleaseDecision(
        blocked=True,
        reasons=[
            "High-risk regression"
        ],
    )

    first_run = create_evaluation_run_record(
        run_id="run-001",
        results=results,
        release_decision=allowed_decision,
    )

    second_run = create_evaluation_run_record(
        run_id="run-002",
        results=results,
        release_decision=blocked_decision,
    )

    append_evaluation_run(
        history_path,
        first_run,
    )

    append_evaluation_run(
        history_path,
        second_run,
    )

    history = load_evaluation_history(
        history_path
    )

    assert len(history) == 2

    assert history[0].run_id == "run-001"
    assert history[0].release_blocked is False

    assert history[1].run_id == "run-002"
    assert history[1].release_blocked is True

def test_gets_metric_history_across_runs():
    first_results = [
        CaseEvaluationResult(
            category="opened_laptop_return",
            risk="high",
            metric_name="FaithfulnessMetric",
            score=1.00,
            threshold=0.90,
        ),
        CaseEvaluationResult(
            category="opened_laptop_return",
            risk="high",
            metric_name="AnswerRelevancyMetric",
            score=0.98,
            threshold=0.80,
        ),
    ]

    second_results = [
        CaseEvaluationResult(
            category="opened_laptop_return",
            risk="high",
            metric_name="FaithfulnessMetric",
            score=0.96,
            threshold=0.90,
        ),
        CaseEvaluationResult(
            category="opened_laptop_return",
            risk="high",
            metric_name="AnswerRelevancyMetric",
            score=0.99,
            threshold=0.80,
        ),
    ]

    allowed_decision = ReleaseDecision(
        blocked=False,
        reasons=[],
    )

    first_run = create_evaluation_run_record(
        run_id="run-001",
        results=first_results,
        release_decision=allowed_decision,
    )

    second_run = create_evaluation_run_record(
        run_id="run-002",
        results=second_results,
        release_decision=allowed_decision,
    )

    history = [
        first_run,
        second_run,
    ]

    metric_history = get_metric_history(
        history=history,
        category="opened_laptop_return",
        metric_name="FaithfulnessMetric",
    )

    assert len(metric_history) == 2

    assert metric_history[0].run_id == "run-001"
    assert metric_history[0].score == 1.00

    assert metric_history[1].run_id == "run-002"
    assert metric_history[1].score == 0.96