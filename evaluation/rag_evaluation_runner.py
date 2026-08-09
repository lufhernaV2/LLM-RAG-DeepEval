from dataclasses import dataclass
from collections.abc import Callable
from deepeval.dataset import Golden
from deepeval.test_case import LLMTestCase

@dataclass
class RAGResult:
    answer: str
    retrieval_context: list[str]

def run_simulated_rag_application(question: str) -> RAGResult:
    normalized_question = question.casefold()

    if "20 days" in normalized_question:
        return RAGResult(
            answer=(
                "Opened laptops may be returned within 30 days. "
                "A 15% restocking fee applies."
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

    if "how will i receive my refund" in normalized_question:
        return RAGResult(
            answer=(
                "Approved refunds are issued to the "
                "original payment method."
            ),
            retrieval_context=[
                (
                    "Approved refunds are issued to the customer's "
                    "original payment method."
                ),
            ],
        )

    if "35 days" in normalized_question:
        return RAGResult(
            answer=(
                "No. Opened laptops must be returned within "
                "30 calendar days of delivery."
            ),
            retrieval_context=[
                (
                    "Northstar Electronics accepts opened laptop returns "
                    "within 30 calendar days of delivery."
                ),
            ],
        )

    return RAGResult(
        answer=(
            "I could not find enough policy information "
            "to answer that question."
        ),
        retrieval_context=[],
    )

def convert_golden_to_test_case(
    golden: Golden,
    result: RAGResult,
) -> LLMTestCase:
    return LLMTestCase(
        input=golden.input,
        actual_output=result.answer,
        expected_output=golden.expected_output,
        retrieval_context=result.retrieval_context,
        additional_metadata=golden.additional_metadata,
    )


def build_test_cases_from_goldens(
    goldens: list[Golden],
) -> list[LLMTestCase]:
    test_cases: list[LLMTestCase] = []

    for golden in goldens:
        result = run_simulated_rag_application(golden.input)

        test_case = convert_golden_to_test_case(
            golden=golden,
            result=result,
        )

        test_cases.append(test_case)

    return test_cases


def run_simulated_rag_application_with_regression(
    question: str,
) -> RAGResult:
    normalized_question = question.casefold()

    if "20 days" in normalized_question:
        return RAGResult(
            answer=(
                "Opened laptops may be returned within 30 days. "
                "There is no restocking fee."
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

    return run_simulated_rag_application(question)

def build_test_cases_with_application(
    goldens: list[Golden],
    rag_application: Callable[[str], RAGResult],
) -> list[LLMTestCase]:
    test_cases: list[LLMTestCase] = []

    for golden in goldens:
        result = rag_application(golden.input)

        test_case = convert_golden_to_test_case(
            golden=golden,
            result=result,
        )

        test_cases.append(test_case)

    return test_cases