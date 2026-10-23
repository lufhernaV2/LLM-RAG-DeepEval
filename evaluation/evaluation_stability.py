from dataclasses import dataclass

BORDERLINE_MARGIN = 0.03

CLEAR_PASS = "clear_pass"
BORDERLINE = "borderline"
CLEAR_FAILURE = "clear_failure"

CONFIRMED_PASS = "confirmed_pass"
CONFIRMED_FAILURE = "confirmed_failure"
UNSTABLE = "unstable"

@dataclass
class EvaluationStabilityResult:
    category: str
    risk: str
    metric_name: str
    initial_score: float
    confirmation_score: float
    threshold: float

    @property
    def initial_passed(self):
        return self.initial_score >= self.threshold

    @property
    def confirmation_passed(self):
        return self.confirmation_score >= self.threshold

    @property
    def outcome(self):
        if self.initial_passed and self.confirmation_passed:
            return CONFIRMED_PASS

        if not self.initial_passed and not self.confirmation_passed:
            return CONFIRMED_FAILURE

        return UNSTABLE

@dataclass
class ConfirmationRequest:
    category: str
    risk: str
    metric_name: str
    initial_score: float
    threshold: float


def classify_result_stability(
    score,
    threshold,
    margin=BORDERLINE_MARGIN,
):
    distance_from_threshold = score - threshold

    if abs(distance_from_threshold) <= margin:
        return BORDERLINE

    if score > threshold:
        return CLEAR_PASS

    return CLEAR_FAILURE

def is_borderline_result(result):
    stability = classify_result_stability(
        score=result.score,
        threshold=result.threshold,
    )

    return stability == BORDERLINE

def get_borderline_results(results):
    return [
        result
        for result in results
        if is_borderline_result(result)
    ]

def build_confirmation_requests(results):
    borderline_results = get_borderline_results(results)

    return [
        ConfirmationRequest(
            category=result.category,
            risk=result.risk,
            metric_name=result.metric_name,
            initial_score=result.score,
            threshold=result.threshold,
        )
        for result in borderline_results
    ]

def create_stability_result(
    confirmation_request,
    confirmation_score,
):
    return EvaluationStabilityResult(
        category=confirmation_request.category,
        risk=confirmation_request.risk,
        metric_name=confirmation_request.metric_name,
        initial_score=confirmation_request.initial_score,
        confirmation_score=confirmation_score,
        threshold=confirmation_request.threshold,
    )

def get_metric_for_confirmation(
    confirmation_request,
    build_metrics_for_risk,
):
    metrics = build_metrics_for_risk(
        confirmation_request.risk
    )

    for metric in metrics:
        if (
            metric.__class__.__name__
            == confirmation_request.metric_name
        ):
            return metric

    raise ValueError(
        "No metric found for confirmation: "
        f"{confirmation_request.metric_name}"
    )

def run_confirmation(
    confirmation_request,
    test_case,
    build_metrics_for_risk,
):
    metric = get_metric_for_confirmation(
        confirmation_request=confirmation_request,
        build_metrics_for_risk=build_metrics_for_risk,
    )

    metric.measure(test_case)

    return create_stability_result(
        confirmation_request=confirmation_request,
        confirmation_score=metric.score,
    )

def should_block_for_stability(stability_results):
    for result in stability_results:
        if result.outcome in {
            CONFIRMED_FAILURE,
            UNSTABLE,
        }:
            return True

    return False

def run_targeted_confirmations(
    current_results,
    test_cases,
    build_metrics_for_risk,
):
    confirmation_requests = build_confirmation_requests(
        current_results
    )

    stability_results = []

    for request in confirmation_requests:
        matching_test_case = None

        for test_case in test_cases:
            category = test_case.additional_metadata["category"]

            if category == request.category:
                matching_test_case = test_case
                break

        if matching_test_case is None:
            raise ValueError(
                "No test case found for confirmation: "
                f"{request.category}"
            )

        stability_result = run_confirmation(
            confirmation_request=request,
            test_case=matching_test_case,
            build_metrics_for_risk=build_metrics_for_risk,
        )

        stability_results.append(stability_result)

    return stability_results