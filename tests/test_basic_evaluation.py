from dotenv import load_dotenv
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
import pytest

load_dotenv()


def test_relevant_pytest_answer() -> None:
    test_case = LLMTestCase(
        input="What is a fixture in pytest?",
        actual_output=(
            "A pytest fixture provides reusable setup data, resources, "
            "or dependencies that tests can request."
        ),
    )

    metric = AnswerRelevancyMetric(
        threshold=0.7,
        model="gpt-4.1-mini",
        include_reason=True,
    )

    assert_test(test_case, [metric])

@pytest.mark.xfail(
        reason="Known irrelevant response should fail the relevancy quality gate",
        strict=True,
)
def test_irrelevant_pytest_answer() -> None:
    test_case = LLMTestCase(
        input="What is a fixture in pytest?",
        actual_output=(
            "Playwright is a browser automation framework used "
            "to test web applications."
        ),
    )

    metric = AnswerRelevancyMetric(
        threshold=0.7,
        model="gpt-4.1-mini",
        include_reason=True,
    )

    assert_test(test_case, [metric])


def test_partially_relevant_pytest_answer() -> None:
    test_case = LLMTestCase(
        input="How do pytest fixtures reduce duplicated setup code?",
        actual_output=(
            "Pytest is a Python testing framework that supports fixtures."
        ),
    )

    metric = AnswerRelevancyMetric(
        threshold=0.7,
        model="gpt-4.1-mini",
        include_reason=True,
    )

    assert_test(test_case, [metric])