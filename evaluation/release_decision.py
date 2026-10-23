from dataclasses import dataclass
from evaluation.evaluation_results import (
    get_failed_results,
    get_high_risk_failures,
    should_block_release,
)

from evaluation.regression_analysis import (
    get_regressions,
    get_high_risk_regressions,
)

from evaluation.evaluation_stability import (
    CONFIRMED_FAILURE,
    UNSTABLE,
)

@dataclass
class ReleaseDecision:
    blocked: bool
    reasons: list[str]

def should_block_release_with_regressions(
    current_results,
    comparisons,
):
    if should_block_release(current_results):
        return True

    if get_high_risk_regressions(comparisons):
        return True

    return False

def build_release_decision(
    current_results,
    comparisons,
    stability_results=None,
):
    reasons = []

    if stability_results is None:
        stability_results = []

    high_risk_failures = get_high_risk_failures(current_results)

    for failure in high_risk_failures:
        reasons.append(
            f"High-risk threshold failure: "
            f"{failure.category} / {failure.metric_name}"
        )

    high_risk_regressions = get_high_risk_regressions(comparisons)

    for regression in high_risk_regressions:
        reasons.append(
            f"High-risk regression: "
            f"{regression.category} / {regression.metric_name}"
        )

    for stability_result in stability_results:
        if stability_result.outcome == CONFIRMED_FAILURE:
            reasons.append(
                f"Confirmed evaluation failure: "
                f"{stability_result.category} / "
                f"{stability_result.metric_name}"
            )

        if stability_result.outcome == UNSTABLE:
            reasons.append(
                f"Unstable evaluation: "
                f"{stability_result.category} / "
                f"{stability_result.metric_name}"
            )

    return ReleaseDecision(
        blocked=len(reasons) > 0,
        reasons=reasons,
    )

def format_release_report(
    current_results,
    comparisons,
    stability_results=None,
):
    if stability_results is None:
        stability_results = []

    failed_results = get_failed_results(current_results)
    high_risk_failures = get_high_risk_failures(current_results)

    regressions = get_regressions(comparisons)
    high_risk_regressions = get_high_risk_regressions(comparisons)

    decision = build_release_decision(
        current_results,
        comparisons,
        stability_results=stability_results,
    )

    release_status = "BLOCKED" if decision.blocked else "ALLOWED"

    lines = [
        "=== AI Evaluation Release Report ===",
        "",
        f"Metric Evaluations: {len(current_results)}",
        f"Threshold Failures: {len(failed_results)}",
        f"High-Risk Threshold Failures: {len(high_risk_failures)}",
        f"Regressions: {len(regressions)}",
        f"High-Risk Regressions: {len(high_risk_regressions)}",
        "",
        f"Release Decision: {release_status}",
    ]

    if decision.reasons:
        lines.extend(
            [
                "",
                "Reasons:",
            ]
        )

        for reason in decision.reasons:
            lines.append(f"- {reason}")

    return "\n".join(lines)