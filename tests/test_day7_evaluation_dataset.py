from datasets.northstar_policy_goldens import (
    northstar_policy_dataset,
)
from deepeval.test_case import LLMTestCase


def test_northstar_dataset_loads() -> None:
    goldens = northstar_policy_dataset.goldens

    print(f"\nGolden count: {len(goldens)}")

    for golden in goldens:
        print(f"\nInput: {golden.input}")
        print(f"Expected output: {golden.expected_output}")
        print(f"Metadata: {golden.additional_metadata}")

    assert len(goldens) == 3

def test_northstar_goldens_have_required_fields() -> None:
    goldens = northstar_policy_dataset.goldens

    for golden in goldens:
        assert golden.input is not None
        assert golden.input.strip() != ""

        assert golden.expected_output is not None
        assert golden.expected_output.strip() != ""

        assert golden.additional_metadata is not None
        assert "category" in golden.additional_metadata
        assert "risk" in golden.additional_metadata


def test_convert_golden_to_test_case() -> None:
    golden = northstar_policy_dataset.goldens[0]

    simulated_actual_output = (
        "Opened laptops may be returned within 30 days. "
        "A 15% restocking fee applies."
    )

    simulated_retrieval_context = [
        (
            "Northstar Electronics accepts opened laptop returns "
            "within 30 calendar days of delivery."
        ),
        (
            "Opened electronics are subject to a 15% "
            "restocking fee."
        ),
    ]

    test_case = LLMTestCase(
        input=golden.input,
        actual_output=simulated_actual_output,
        expected_output=golden.expected_output,
        retrieval_context=simulated_retrieval_context,
    )

    assert test_case.input == golden.input
    assert test_case.expected_output == golden.expected_output
    assert test_case.actual_output == simulated_actual_output
    assert test_case.retrieval_context == simulated_retrieval_context

    print(f"\nInput: {test_case.input}")
    print(f"Actual output: {test_case.actual_output}")
    print(f"Expected output: {test_case.expected_output}")
    print(f"Retrieval context: {test_case.retrieval_context}")