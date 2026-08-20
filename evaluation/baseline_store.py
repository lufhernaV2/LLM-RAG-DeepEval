from evaluation.evaluation_results import CaseEvaluationResult
from evaluation.evaluation_results import should_block_release
import json

def result_to_dict(result):
    return {
        "category": result.category,
        "risk": result.risk,
        "metric_name": result.metric_name,
        "score": result.score,
        "threshold": result.threshold,
    }

def dict_to_result(data):
    return CaseEvaluationResult(
        category=data["category"],
        risk=data["risk"],
        metric_name=data["metric_name"],
        score=data["score"],
        threshold=data["threshold"],
    )

def save_baseline(results, file_path):
    serialized_results = [
        result_to_dict(result)
        for result in results
    ]

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(
            serialized_results,
            file,
            indent=2,
        )

def load_baseline(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        serialized_results = json.load(file)

    return [
        dict_to_result(data)
        for data in serialized_results
    ]

def can_promote_to_baseline(results):
    return not should_block_release(results)