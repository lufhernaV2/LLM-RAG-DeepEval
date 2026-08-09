from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

EVALUATOR_MODEL = "gpt-4.1-mini"
PASSING_THRESHOLD = 0.7

def create_answer_relevancy_metric() -> AnswerRelevancyMetric:
    return AnswerRelevancyMetric(
        threshold=PASSING_THRESHOLD,
        model=EVALUATOR_MODEL,
        include_reason=True,
    )


def create_faithfulness_metric() -> FaithfulnessMetric:
    return FaithfulnessMetric(
        threshold=PASSING_THRESHOLD,
        model=EVALUATOR_MODEL,
        include_reason=True,
    )

def evaluate_test_case_batch(
    test_cases: list[LLMTestCase],
) -> None:
    metrics = [
        create_answer_relevancy_metric(),
        create_faithfulness_metric(),
    ]

    evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(
            run_async=False,
        ),
    )