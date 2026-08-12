from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric

EVALUATOR_MODEL = "gpt-4.1-mini"
EVALUATION_THRESHOLDS = {
    "high": {
        "answer_relevancy": 0.80,
        "faithfulness": 0.90,
    },
    "medium": {
        "answer_relevancy": 0.70,
        "faithfulness": 0.70,
    },
    "low": {
        "answer_relevancy": 0.60,
        "faithfulness": 0.60,
    },
}


def get_evaluation_thresholds(risk):
    return EVALUATION_THRESHOLDS[risk]


def get_faithfulness_threshold(risk):
    return EVALUATION_THRESHOLDS[risk]["faithfulness"]


def build_metrics_for_risk(risk):
    thresholds = get_evaluation_thresholds(risk)

    return [
        AnswerRelevancyMetric(
            threshold=thresholds["answer_relevancy"],
            model=EVALUATOR_MODEL,
            include_reason=True,
        ),
        FaithfulnessMetric(
            threshold=thresholds["faithfulness"],
            model=EVALUATOR_MODEL,
            include_reason=True,
        ),
    ]