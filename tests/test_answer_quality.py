from dotenv import load_dotenv

from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

import pytest

load_dotenv()


correctness_metric = GEval(
    name="Answer Correctness",
    criteria=(
        "Evaluate only factual correctness. Compare the actual output with "
        "the expected output and determine whether its claims are true. "
        "Do not penalize an answer merely because it is brief or incomplete. "
        "A factually true but partial answer may still be correct. "
        "Penalize contradictions, false mechanisms, invented claims, and "
        "materially misleading statements."
    ),
    evaluation_params=[
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.EXPECTED_OUTPUT,
    ],
    threshold=0.7,
    model="gpt-4.1-mini",
)

completeness_metric = GEval(
    name="Answer Completeness",
    criteria=(
        "Evaluate whether the actual output includes all essential information "
        "needed to answer the input. Compare it with the expected output to "
        "identify missing concepts. Focus only on coverage and sufficiency. "
        "Do not reduce the completeness score solely because a claim is "
        "factually incorrect; factual accuracy is evaluated separately."
    ),
    evaluation_params=[
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.EXPECTED_OUTPUT,
    ],
    threshold=0.7,
    model="gpt-4.1-mini",
)

def test_complete_and_correct_fixture_answer() -> None:
    test_case = LLMTestCase(
        input="How do pytest fixtures reduce duplicated setup code?",
        actual_output=(
            "Fixtures place reusable setup logic in one function. "
            "Multiple tests can request the same fixture instead of "
            "repeating that setup inside every test."
        ),
        expected_output=(
            "Pytest fixtures reduce duplication by centralizing reusable "
            "setup logic. Tests request the fixture and share the setup "
            "instead of implementing the same preparation repeatedly."
        ),
    )

    assert_test(
        test_case=test_case,
        metrics=[
            correctness_metric,
            completeness_metric,
        ],
    )

@pytest.mark.xfail(
    reason="Incomplete response should fail at least one answer-quality gate",
    strict=True,
)
def test_incomplete_fixture_answer() -> None:
    test_case = LLMTestCase(
        input="How do pytest fixtures reduce duplicated setup code?",
        actual_output="Pytest supports reusable fixtures.",
        expected_output=(
            "Pytest fixtures reduce duplication by centralizing reusable "
            "setup logic. Tests request the fixture and share the setup "
            "instead of implementing the same preparation repeatedly."
        ),
    )

    assert_test(
        test_case=test_case,
        metrics=[
            correctness_metric,
            completeness_metric,
        ],
    )

@pytest.mark.xfail(
    reason="Factually incorrect response should fail the correctness gate",
    strict=True,
)
def test_incorrect_fixture_answer() -> None:
    test_case = LLMTestCase(
        input="How do pytest fixtures reduce duplicated setup code?",
        actual_output=(
            "Pytest fixtures reduce duplication by automatically copying "
            "the setup code into every test before execution."
        ),
        expected_output=(
            "Pytest fixtures reduce duplication by centralizing reusable "
            "setup logic. Tests request the fixture and share the setup "
            "instead of implementing the same preparation repeatedly."
        ),
    )

    assert_test(
        test_case=test_case,
        metrics=[
            correctness_metric,
            completeness_metric,
        ],
    )