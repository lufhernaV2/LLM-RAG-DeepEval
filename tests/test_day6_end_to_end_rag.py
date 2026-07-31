import re
from typing import Any

from deepeval.metrics import (
    AnswerRelevancyMetric,
    BaseMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from deepeval.metrics.g_eval import Rubric


EVALUATOR_MODEL = "gpt-4.1-mini"
PASSING_THRESHOLD = 0.7

class MaterialClaimGroundednessMetric(BaseMetric):
    def __init__(
        self,
        material_topic_aliases: dict[str, list[str]],
        threshold: float = 1.0,
    ) -> None:
        self.threshold = threshold

        self.material_topic_aliases = {
            topic_name: [
                alias.casefold().strip()
                for alias in aliases
            ]
            for topic_name, aliases
            in material_topic_aliases.items()
        }

        self.score: float | None = None
        self.reason: str | None = None
        self.success: bool = False
        self.error: str | None = None

        self.include_reason = True
        self.strict_mode = True
        self.async_mode = False
        self.evaluation_model = "deterministic-python"

    def measure(self, test_case: LLMTestCase) -> float:
        try:
            retrieval_context = test_case.retrieval_context or []

            actual_output = self._normalize_text(
                test_case.actual_output
            )
            combined_context = self._normalize_text(
                " ".join(retrieval_context)
            )

            output_sentences = self._split_sentences(actual_output)

            unsupported_claims: list[tuple[str, str]] = []

            for sentence in output_sentences:
                for topic_name, aliases in (
                    self.material_topic_aliases.items()
                ):
                    output_mentions_topic = any(
                        alias in sentence
                        for alias in aliases
                    )

                    if not output_mentions_topic:
                        continue

                    context_mentions_topic = any(
                        alias in combined_context
                        for alias in aliases
                    )

                    if not context_mentions_topic:
                        unsupported_claims.append(
                            (topic_name, sentence)
                        )

            if unsupported_claims:
                self.score = 0.0
                self.success = False

                formatted_claims = "; ".join(
                    (
                        f'{topic_name}: "{claim}"'
                    )
                    for topic_name, claim
                    in unsupported_claims
                )

                self.reason = (
                    "The response contains material policy claims "
                    "whose concepts are absent from the retrieval "
                    f"context: {formatted_claims}"
                )
            else:
                self.score = 1.0
                self.success = True
                self.reason = (
                    "Every generated claim involving a configured "
                    "material policy concept has corresponding "
                    "concept evidence in the retrieval context."
                )

            return self.score

        except Exception as exc:
            self.error = str(exc)
            self.score = 0.0
            self.success = False
            self.reason = (
                "The deterministic groundedness evaluation failed: "
                f"{exc}"
            )
            raise

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            self.success = bool(
                self.score is not None
                and self.score >= self.threshold
            )

        return self.success

    @property
    def __name__(self) -> str:
        return "Material Claim Groundedness"

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.casefold().split())

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        return [
            sentence.strip()
            for sentence in re.split(r"[.!?]+", text)
            if sentence.strip()
        ]

    
def create_material_claim_groundedness_metric(
    ) -> MaterialClaimGroundednessMetric:
        return MaterialClaimGroundednessMetric(
            material_topic_aliases={
                "restocking fee policy": [
                    "restocking fee",
                    "return fee",
                    "return charge",
                    "return processing fee",
                    "return penalty",
                ],
            },
            threshold=1.0,
        )


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
        verbose_mode=True,
    )


def create_contextual_precision_metric() -> ContextualPrecisionMetric:
    return ContextualPrecisionMetric(
        threshold=PASSING_THRESHOLD,
        model=EVALUATOR_MODEL,
        include_reason=True,
    )


def create_contextual_recall_metric() -> ContextualRecallMetric:
    return ContextualRecallMetric(
        threshold=PASSING_THRESHOLD,
        model=EVALUATOR_MODEL,
        include_reason=True,
    )


def create_contextual_relevancy_metric() -> ContextualRelevancyMetric:
    return ContextualRelevancyMetric(
        threshold=PASSING_THRESHOLD,
        model=EVALUATOR_MODEL,
        include_reason=True,
    )

def measure_and_print(
    metric_name: str,
    metric: Any,
    test_case: LLMTestCase,
) -> None:
    metric.measure(test_case)

    passed = metric.score >= metric.threshold

    print(f"\n{metric_name}")
    print(f"Score: {metric.score}")
    print(f"Threshold: {metric.threshold}")
    print(f"Passed: {passed}")
    print(f"Reason: {metric.reason}")


def measure_print_and_assert(
        metric_name: str,
        metric: Any,
        test_case: LLMTestCase,
    ) -> None:
    metric.measure(test_case)

    passed = metric.score >= metric.threshold

    print(f"\n{metric_name}")
    print(f"Score: {metric.score}")
    print(f"Threshold: {metric.threshold}")
    print(f"Passed: {passed}")
    print(f"Reason: {metric.reason}")

    assert passed, (
        f"{metric_name} failed with score {metric.score}. "
        f"Reason: {metric.reason}"
    )


def test_healthy_end_to_end_rag_pipeline() -> None:
    test_case = LLMTestCase(
        input=(
            "I opened a laptop 20 days ago. Can I return it, "
            "is there a fee, and how will I receive my refund?"
        ),
        actual_output=(
            "Yes. Opened laptops may be returned within 30 days. "
            "A 15% restocking fee applies. If the return is approved, "
            "the refund will be issued to your original payment method."
        ),
        expected_output=(
            "Opened laptops may be returned within 30 days. "
            "Opened laptop returns are subject to a 15% restocking fee. "
            "Approved refunds are issued to the original payment method."
        ),
        retrieval_context=[
            (
                "Northstar Electronics accepts opened laptop returns "
                "within 30 calendar days of delivery."
            ),
            (
                "Opened electronics returned in otherwise eligible "
                "condition are subject to a 15% restocking fee."
            ),
            (
                "Approved refunds are issued to the customer's "
                "original payment method."
            ),
        ],
    )

    measure_and_print(
        metric_name="Material Claim Groundedness",
        metric=create_material_claim_groundedness_metric(),
        test_case=test_case,
    )


def test_missing_evidence_honest_generator() -> None:
    test_case = LLMTestCase(
        input=(
            "I opened a laptop 20 days ago. Can I return it, "
            "is there a fee, and how will I receive my refund?"
        ),
        actual_output=(
            "Opened laptops may be returned within 30 days. "
            "The retrieved policy does not specify whether a restocking "
            "fee applies. If the return is approved, the refund will be "
            "issued to your original payment method."
        ),
        expected_output=(
            "Opened laptops may be returned within 30 days. "
            "Opened laptop returns are subject to a 15% restocking fee. "
            "Approved refunds are issued to the original payment method."
        ),
        retrieval_context=[
            (
                "Northstar Electronics accepts opened laptop returns "
                "within 30 calendar days of delivery."
            ),
            (
                "Approved refunds are issued to the customer's "
                "original payment method."
            ),
        ],
    )

    measure_and_print(
        metric_name="Faithfulness",
        metric=create_faithfulness_metric(),
        test_case=test_case,
    )

def test_missing_evidence_generator_invents_answer() -> None:
    test_case = LLMTestCase(
        input=(
            "I opened a laptop 20 days ago. Can I return it, "
            "is there a fee, and how will I receive my refund?"
        ),
        actual_output=(
            "Opened laptops may be returned within 30 days. "
            "There is no restocking fee. If the return is approved, "
            "the refund will be issued to your original payment method."
        ),
        expected_output=(
            "Opened laptops may be returned within 30 days. "
            "Opened laptop returns are subject to a 15% restocking fee. "
            "Approved refunds are issued to the original payment method."
        ),
        retrieval_context=[
            (
                "Northstar Electronics accepts opened laptop returns "
                "within 30 calendar days of delivery."
            ),
            (
                "Approved refunds are issued to the customer's "
                "original payment method."
            ),
        ]
    )

    measure_and_print(
        metric_name="Material Claim Groundedness",
        metric=create_material_claim_groundedness_metric(),
        test_case=test_case,
    )

def test_material_claim_groundedness_detects_configured_paraphrase() -> None:
    test_case = LLMTestCase(
        input=(
            "I opened a laptop 20 days ago. Can I return it, "
            "is there a fee, and how will I receive my refund?"
        ),
        actual_output=(
            "Opened laptops may be returned within 30 days. "
            "There is no return charge. "
            "If the return is approved, the refund will be "
            "issued to your original payment method."
        ),
        expected_output=(
            "Opened laptops may be returned within 30 days. "
            "Opened laptop returns are subject to a 15% restocking fee. "
            "Approved refunds are issued to the original payment method."
        ),
        retrieval_context=[
            (
                "Northstar Electronics accepts opened laptop returns "
                "within 30 calendar days of delivery."
            ),
            (
                "Approved refunds are issued to the customer's "
                "original payment method."
            ),
        ],
    )

    measure_and_print(
        metric_name="Material Claim Groundedness",
        metric=create_material_claim_groundedness_metric(),
        test_case=test_case,
    )


def test_material_claim_groundedness_supports_alias_in_context() -> None:
    test_case = LLMTestCase(
        input=(
            "I opened a laptop 20 days ago. Can I return it, "
            "and will I have to pay anything?"
        ),
        actual_output=(
            "Opened laptops may be returned within 30 days. "
            "A return charge of 15% applies."
        ),
        expected_output=(
            "Opened laptops may be returned within 30 days. "
            "Opened laptop returns are subject to a 15% restocking fee."
        ),
        retrieval_context=[
            (
                "Northstar Electronics accepts opened laptop returns "
                "within 30 calendar days of delivery."
            ),
            (
                "Opened electronics are subject to a 15% "
                "restocking fee."
            ),
        ],
    )

    measure_and_print(
        metric_name="Material Claim Groundedness",
        metric=create_material_claim_groundedness_metric(),
        test_case=test_case,
    )
    
# Calibration finding:
# GEval consistently identified the unsupported fee claim in its reason,
# but did not reliably assign a failing numerical score.
# Do not use this metric as a deterministic CI quality gate.

def create_strict_groundedness_metric() -> GEval:
    return GEval(
        name="Strict Groundedness",
        criteria=(
            "Evaluate whether every material factual claim in the actual "
            "output is explicitly supported by the retrieval context. "
            "Silence in the retrieval context is not support."
        ),
        evaluation_steps=[
            (
                "Extract every material factual claim from the actual output "
                "as a separate atomic claim."
            ),
            (
                "For each claim, identify explicit evidence in the retrieval "
                "context that supports it."
            ),
            (
                "Mark a claim unsupported when no explicit supporting "
                "evidence exists, even when the context does not contradict it."
            ),
            (
                "Determine the final score using the supplied rubric."
            ),
        ],
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        rubric=[
            Rubric(
                score_range=(0, 4),
                expected_outcome=(
                    "At least one material factual claim is contradicted by "
                    "the retrieval context."
                ),
            ),
            Rubric(
                score_range=(5, 6),
                expected_outcome=(
                    "At least one material factual claim is unsupported "
                    "because the retrieval context is silent about it."
                ),
            ),
            Rubric(
                score_range=(7, 8),
                expected_outcome=(
                    "All material factual claims are supported, but some "
                    "support requires reasonable interpretation."
                ),
            ),
            Rubric(
                score_range=(9, 10),
                expected_outcome=(
                    "Every material factual claim is directly and explicitly "
                    "supported by the retrieval context."
                ),
            ),
        ],
        threshold=0.7,
        model=EVALUATOR_MODEL,
    )