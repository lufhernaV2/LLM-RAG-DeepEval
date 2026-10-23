from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

from evaluation.baseline_store import (
    dict_to_result,
    result_to_dict,
)


import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from evaluation.baseline_store import (
    dict_to_result,
    result_to_dict,
)

@dataclass
class EvaluationRunRecord:
    run_id: str
    timestamp: str
    results: list
    release_blocked: bool


@dataclass
class HistoricalMetricScore:
    run_id: str
    timestamp: str
    category: str
    metric_name: str
    score: float
    threshold: float

def create_evaluation_run_record(
    run_id,
    results,
    release_decision,
):
    return EvaluationRunRecord(
        run_id=run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        results=results,
        release_blocked=release_decision.blocked,
    )

def run_record_to_dict(run_record):
    return {
        "run_id": run_record.run_id,
        "timestamp": run_record.timestamp,
        "release_blocked": run_record.release_blocked,
        "results": [
            result_to_dict(result)
            for result in run_record.results
        ],
    }


def dict_to_run_record(data):
    return EvaluationRunRecord(
        run_id=data["run_id"],
        timestamp=data["timestamp"],
        release_blocked=data["release_blocked"],
        results=[
            dict_to_result(result_data)
            for result_data in data["results"]
        ],
    )

def load_evaluation_history(path):
    path = Path(path)

    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return [
        dict_to_run_record(run_data)
        for run_data in data
    ]


def save_evaluation_history(path, history):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = [
        run_record_to_dict(run_record)
        for run_record in history
    ]

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
        )

def append_evaluation_run(path, run_record):
    history = load_evaluation_history(path)

    history.append(run_record)

    save_evaluation_history(
        path,
        history,
    )

def get_metric_history(
    history,
    category,
    metric_name,
):
    metric_history = []

    for run_record in history:
        for result in run_record.results:
            if (
                result.category == category
                and result.metric_name == metric_name
            ):
                metric_history.append(
                    HistoricalMetricScore(
                        run_id=run_record.run_id,
                        timestamp=run_record.timestamp,
                        category=result.category,
                        metric_name=result.metric_name,
                        score=result.score,
                        threshold=result.threshold,
                    )
                )

    return metric_history