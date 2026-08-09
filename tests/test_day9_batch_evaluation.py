from datasets.northstar_policy_goldens import (
    northstar_policy_dataset,
)
from evaluation.dataset_metric_runner import (
    evaluate_test_case_batch,
)
from evaluation.rag_evaluation_runner import (
    build_test_cases_from_goldens,
)

from evaluation.rag_evaluation_runner import (
    build_test_cases_from_goldens,
    build_test_cases_with_application,
    run_simulated_rag_application_with_regression,
)

def test_evaluate_all_northstar_cases() -> None:
    test_cases = build_test_cases_from_goldens(
        northstar_policy_dataset.goldens
    )

    print(f"\nEvaluating {len(test_cases)} test cases")

    evaluate_test_case_batch(test_cases)

def test_evaluate_first_northstar_case() -> None:
    test_cases = build_test_cases_from_goldens(
        northstar_policy_dataset.goldens
    )

    first_test_case = test_cases[0]

    evaluate_test_case_batch(
        [first_test_case]
    )

def test_batch_evaluation_detects_regression() -> None:
    test_cases = build_test_cases_with_application(
        goldens=northstar_policy_dataset.goldens,
        rag_application=run_simulated_rag_application_with_regression,
    )

    print(
        f"\nEvaluating {len(test_cases)} test cases "
        "with controlled regression"
    )

    evaluate_test_case_batch(test_cases)