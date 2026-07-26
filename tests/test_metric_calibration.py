from dotenv import load_dotenv

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

load_dotenv()

def create_correctness_metric() -> GEval:
    return GEval(
        name="Answer Correctness",
        evaluation_steps=[
            "List the factual claims made in the actual output.",
            "Check each factual claim against the expected output.",
            "Do not penalize the actual output for omitted information.",
            "Do not evaluate completeness, detail, or explanatory depth.",
            "Assign a high score when every stated claim is true, even if the answer is brief.",
            "Lower the score only for false, contradictory, invented, or misleading claims.",
        ],
        evaluation_params=[
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        threshold=0.7,
        model="gpt-4.1-mini",
    )


def create_completeness_metric() -> GEval:
    return GEval(
        name="Answer Completeness",
        evaluation_steps=[
            "Identify the essential concepts present in the expected output.",
            "Determine which essential concepts are present in the actual output.",
            "Evaluate only coverage of the required concepts.",
            "Do not penalize factual inaccuracies, contradictions, or misleading claims.",
            "Do not evaluate correctness, truthfulness, or clarity.",
            "Assign a high score when all essential concepts are mentioned, even if one or more claims are factually wrong.",
            "Lower the score only when essential concepts are missing.",
        ],
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        threshold=0.7,
        model="gpt-4.1-mini",
    )


def test_strong_answer_passes_both_metrics() -> None:
    test_case = LLMTestCase(
        input="How do pytest fixtures reduce duplicated setup code?",
        actual_output=(
            "Fixtures centralize reusable setup logic in one function. "
            "Multiple tests request and share the same setup instead of "
            "repeating the same preparation code."
        ),
        expected_output=(
            "Pytest fixtures reduce duplication by centralizing reusable "
            "setup logic. Tests request the fixture and share the setup "
            "instead of implementing the same preparation repeatedly."
        ),
    )

    correctness_score = measure(
        create_correctness_metric(),
        test_case,
    )
    completeness_score = measure(
        create_completeness_metric(),
        test_case,
    )

    assert correctness_score >= 0.7, (
        "A strong answer should pass factual correctness."
    )
    assert completeness_score >= 0.7, (
        "A strong answer should include all essential concepts."
    )

def test_true_but_incomplete_answer_is_scored_separately() -> None:
    test_case = LLMTestCase(
        input="How do pytest fixtures reduce duplicated setup code?",
        actual_output="Pytest fixtures are reusable.",
        expected_output=(
            "Pytest fixtures reduce duplication by centralizing reusable "
            "setup logic. Tests request the fixture and share the setup "
            "instead of implementing the same preparation repeatedly."
        ),
    )

    correctness_score = measure(
        create_correctness_metric(),
        test_case,
    )
    completeness_score = measure(
        create_completeness_metric(),
        test_case,
    )

    assert correctness_score >= 0.6, (
        "A true but brief statement should not fail factual correctness."
    )
    assert completeness_score < 0.7, (
        "The answer should fail completeness because it does not explain "
        "how fixtures reduce duplication."
    )

def test_detailed_but_false_answer_fails_correctness() -> None:
    test_case = LLMTestCase(
        input="How do pytest fixtures reduce duplicated setup code?",
        actual_output=(
            "Fixtures centralize setup logic in a reusable function that "
            "multiple tests can request. Pytest then permanently copies that "
            "fixture code into each test file before execution."
        ),
        expected_output=(
            "Pytest fixtures reduce duplication by centralizing reusable "
            "setup logic. Tests request the fixture and share the setup "
            "instead of implementing the same preparation repeatedly."
        ),
    )

    correctness_score = measure(
        create_correctness_metric(),
        test_case,
    )
    completeness_score = measure(
        create_completeness_metric(),
        test_case,
    )

    assert correctness_score < 0.7, (
        "The answer should fail because pytest does not permanently copy "
        "fixture code into each test file."
    )
    assert completeness_score >= 0.7, (
        "The answer covers the centralization, reuse, and test-request "
        "concepts, even though one factual claim is false."
    )


def measure(metric: GEval, test_case: LLMTestCase) -> float:
    metric.measure(test_case)

    assert metric.score is not None, (
        f"{metric.name} did not produce a score. "
        f"Reason: {metric.reason}"
    )

    print(
        f"\n{metric.name}"
        f"\nScore: {metric.score}"
        f"\nReason: {metric.reason}"
    )

    return metric.score