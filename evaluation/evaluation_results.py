from dataclasses import dataclass


@dataclass
class CaseEvaluationResult:
    category: str
    risk: str
    metric_name: str
    score: float
    threshold: float

    @property
    def passed(self):
        return self.score >= self.threshold


def create_case_evaluation_result(
    category,
    risk,
    metric,
):
    return CaseEvaluationResult(
        category=category,
        risk=risk,
        metric_name=metric.__class__.__name__,
        score=metric.score,
        threshold=metric.threshold,
    )

def create_results_for_metrics(
    category,
    risk,
    test_case,
    metrics,
):
    results = []

    for metric in metrics:
        metric.measure(test_case)

        result = create_case_evaluation_result(
            category=category,
            risk=risk,
            metric=metric,
        )

        results.append(result)

    return results


def evaluate_dataset_to_results(
    goldens,
    test_cases,
    build_metrics_for_risk,
):
    results = []

    for golden, test_case in zip(goldens, test_cases):
        category = golden.additional_metadata["category"]
        risk = golden.additional_metadata["risk"]

        metrics = build_metrics_for_risk(risk)

        case_results = create_results_for_metrics(
            category=category,
            risk=risk,
            test_case=test_case,
            metrics=metrics,
        )

        results.extend(case_results)

    return results


def get_failed_results(results):
    return [
        result
        for result in results
        if not result.passed
    ]


def get_high_risk_failures(results):
    return [
        result
        for result in results
        if not result.passed and result.risk == "high"
    ]


def should_block_release(results):
    high_risk_failures = get_high_risk_failures(results)

    return len(high_risk_failures) > 0

def build_evaluation_summary(results):
    failed_results = get_failed_results(results)
    high_risk_failures = get_high_risk_failures(results)

    total_results = len(results)
    failed_count = len(failed_results)
    passed_count = total_results - failed_count

    release_decision = (
        "BLOCKED"
        if should_block_release(results)
        else "ALLOWED"
    )

    return {
        "total_results": total_results,
        "passed": passed_count,
        "failed": failed_count,
        "high_risk_failures": len(high_risk_failures),
        "release_decision": release_decision,
    }

def format_evaluation_summary(results):
    summary = build_evaluation_summary(results)
    failed_results = get_failed_results(results)

    lines = [
        "=== AI Evaluation Release Summary ===",
        "",
        f"Metric Evaluations: {summary['total_results']}",
        f"Passed: {summary['passed']}",
        f"Failed: {summary['failed']}",
        f"High-Risk Failures: {summary['high_risk_failures']}",
        "",
        f"Release Decision: {summary['release_decision']}",
    ]

    if failed_results:
        lines.append("")
        lines.append("Failures:")

        for result in failed_results:
            lines.extend(
                [
                    f"- Category: {result.category}",
                    f"  Risk: {result.risk}",
                    f"  Metric: {result.metric_name}",
                    f"  Score: {result.score:.2f}",
                    f"  Required: {result.threshold:.2f}",
                ]
            )

    return "\n".join(lines)