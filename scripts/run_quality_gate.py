import sys

from datasets.northstar_policy_goldens import northstar_policy_goldens
from evaluation.evaluation_pipeline import (
    get_pipeline_exit_code,
    run_evaluation_pipeline,
)
from evaluation.evaluation_policy import build_metrics_for_risk
from evaluation.evaluation_results import evaluate_dataset_to_results
from evaluation.rag_evaluation_runner import build_test_cases_from_goldens


BASELINE_PATH = "baselines/northstar_baseline.json"


def main():
    test_cases = build_test_cases_from_goldens(
        northstar_policy_goldens
    )

    current_results = evaluate_dataset_to_results(
        goldens=northstar_policy_goldens,
        test_cases=test_cases,
        build_metrics_for_risk=build_metrics_for_risk,
    )

    pipeline_result = run_evaluation_pipeline(
        current_results=current_results,
        baseline_path=BASELINE_PATH,
    )

    print()
    print(pipeline_result.report)

    return get_pipeline_exit_code(pipeline_result)


if __name__ == "__main__":
    sys.exit(main())