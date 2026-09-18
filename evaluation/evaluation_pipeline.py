from dataclasses import dataclass

from evaluation.baseline_store import load_baseline
from evaluation.regression_analysis import compare_evaluation_runs
from evaluation.release_decision import (
    build_release_decision,
    format_release_report,
)


@dataclass
class EvaluationPipelineResult:
    current_results: list
    comparisons: list
    release_decision: object
    report: str


def run_evaluation_pipeline(
    current_results,
    baseline_path,
):
    baseline_results = load_baseline(baseline_path)

    comparisons = compare_evaluation_runs(
        baseline_results,
        current_results,
    )

    decision = build_release_decision(
        current_results,
        comparisons,
    )

    report = format_release_report(
        current_results,
        comparisons,
    )

    return EvaluationPipelineResult(
        current_results=current_results,
        comparisons=comparisons,
        release_decision=decision,
        report=report,
    )

def get_pipeline_exit_code(pipeline_result):
    if pipeline_result.release_decision.blocked:
        return 1

    return 0