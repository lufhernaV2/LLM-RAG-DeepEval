from dataclasses import dataclass


REGRESSION_TOLERANCE = 0.05


@dataclass
class EvaluationComparison:
    category: str
    risk: str
    metric_name: str
    baseline_score: float
    current_score: float

    @property
    def delta(self):
        return self.current_score - self.baseline_score

    @property
    def regressed(self):
        return self.delta < -REGRESSION_TOLERANCE


def calculate_score_delta(baseline_result, current_result):
    return current_result.score - baseline_result.score

def create_evaluation_comparison(
    baseline_result,
    current_result,
):
    return EvaluationComparison(
        category=current_result.category,
        risk=current_result.risk,
        metric_name=current_result.metric_name,
        baseline_score=baseline_result.score,
        current_score=current_result.score,
    )

def get_result_key(result):
    return (
        result.category,
        result.metric_name,
    )

def compare_evaluation_runs(
    baseline_results,
    current_results,
):
    baseline_by_key = {
        get_result_key(result): result
        for result in baseline_results
    }

    comparisons = []

    for current_result in current_results:
        key = get_result_key(current_result)

        baseline_result = baseline_by_key[key]

        comparison = create_evaluation_comparison(
            baseline_result,
            current_result,
        )

        comparisons.append(comparison)

    return comparisons

def get_regressions(comparisons):
    return [
        comparison
        for comparison in comparisons
        if comparison.regressed
    ]


def get_high_risk_regressions(comparisons):
    return [
        comparison
        for comparison in comparisons
        if comparison.regressed and comparison.risk == "high"
    ]

def build_regression_summary(comparisons):
    regressions = get_regressions(comparisons)
    high_risk_regressions = get_high_risk_regressions(comparisons)

    largest_regression = None

    if regressions:
        largest_regression = min(
            regressions,
            key=lambda comparison: comparison.delta,
        )

    return {
        "total_comparisons": len(comparisons),
        "regressions": len(regressions),
        "high_risk_regressions": len(high_risk_regressions),
        "largest_regression": largest_regression,
    }

def format_regression_summary(comparisons):
    summary = build_regression_summary(comparisons)
    regressions = get_regressions(comparisons)

    lines = [
        "=== AI Evaluation Regression Summary ===",
        "",
        f"Metric Comparisons: {summary['total_comparisons']}",
        f"Regressions: {summary['regressions']}",
        f"High-Risk Regressions: {summary['high_risk_regressions']}",
    ]

    largest_regression = summary["largest_regression"]

    if largest_regression is not None:
        lines.extend(
            [
                "",
                "Largest Regression:",
                f"Category: {largest_regression.category}",
                f"Risk: {largest_regression.risk}",
                f"Metric: {largest_regression.metric_name}",
                f"Baseline: {largest_regression.baseline_score:.2f}",
                f"Current: {largest_regression.current_score:.2f}",
                f"Delta: {largest_regression.delta:.2f}",
            ]
        )

    if regressions:
        lines.append("")
        lines.append("All Regressions:")

        for comparison in regressions:
            lines.extend(
                [
                    f"- {comparison.category} | {comparison.metric_name}",
                    f"  Risk: {comparison.risk}",
                    f"  Baseline: {comparison.baseline_score:.2f}",
                    f"  Current: {comparison.current_score:.2f}",
                    f"  Delta: {comparison.delta:.2f}",
                ]
            )

    return "\n".join(lines)