from datasets.northstar_policy_goldens import (
    northstar_policy_dataset,
)
from evaluation.rag_evaluation_runner import (
    build_test_cases_from_goldens,
)

def test_builds_test_case_for_every_golden() -> None:
    goldens = northstar_policy_dataset.goldens

    test_cases = build_test_cases_from_goldens(goldens)

    print(f"\nGolden count: {len(goldens)}")
    print(f"Test case count: {len(test_cases)}")

    for test_case in test_cases:
        print(f"\nInput: {test_case.input}")
        print(f"Actual output: {test_case.actual_output}")
        print(f"Expected output: {test_case.expected_output}")
        print(
            f"Retrieval context: "
            f"{test_case.retrieval_context}"
        )

    assert len(test_cases) == len(goldens)


def test_generated_test_cases_have_runtime_fields() -> None:
    test_cases = build_test_cases_from_goldens(
        northstar_policy_dataset.goldens
    )

    for test_case in test_cases:
        assert test_case.input is not None
        assert test_case.input.strip() != ""

        assert test_case.actual_output is not None
        assert test_case.actual_output.strip() != ""

        assert test_case.expected_output is not None
        assert test_case.expected_output.strip() != ""

        assert test_case.retrieval_context is not None
        assert len(test_case.retrieval_context) > 0


def test_golden_metadata_is_preserved() -> None:
    test_cases = build_test_cases_from_goldens(
        northstar_policy_dataset.goldens
    )

    for test_case in test_cases:
        assert test_case.additional_metadata is not None
        assert "category" in test_case.additional_metadata
        assert "risk" in test_case.additional_metadata

        print(
            f"\nInput: {test_case.input}"
            f"\nMetadata: {test_case.additional_metadata}"
        )


def test_generated_test_cases_have_runtime_fields() -> None:
    test_cases = build_test_cases_from_goldens(
        northstar_policy_dataset.goldens
    )

    for test_case in test_cases:
        assert test_case.input is not None
        assert test_case.input.strip() != ""

        assert test_case.actual_output is not None
        assert test_case.actual_output.strip() != ""

        assert test_case.expected_output is not None
        assert test_case.expected_output.strip() != ""

        assert test_case.retrieval_context is not None
        assert len(test_case.retrieval_context) > 0


def test_golden_metadata_is_preserved() -> None:
    test_cases = build_test_cases_from_goldens(
        northstar_policy_dataset.goldens
    )

    for test_case in test_cases:
        assert test_case.additional_metadata is not None
        assert "category" in test_case.additional_metadata
        assert "risk" in test_case.additional_metadata

        print(f"\nInput: {test_case.input}")
        print(f"Metadata: {test_case.additional_metadata}")