from deepeval.dataset import EvaluationDataset, Golden


northstar_policy_goldens = [
    Golden(
        input=(
            "I opened a laptop 20 days ago. "
            "Can I return it and is there a fee?"
        ),
        expected_output=(
            "Opened laptops may be returned within 30 days. "
            "A 15% restocking fee applies."
        ),
        additional_metadata={
            "category": "opened_laptop_return",
            "risk": "high",
        },
    ),
    Golden(
        input=(
            "How will I receive my refund after "
            "an approved laptop return?"
        ),
        expected_output=(
            "Approved refunds are issued to the "
            "original payment method."
        ),
        additional_metadata={
            "category": "refund_method",
            "risk": "medium",
        },
    ),
    Golden(
        input=(
            "Can I return an opened laptop 35 days "
            "after it was delivered?"
        ),
        expected_output=(
            "No. Opened laptops must be returned "
            "within 30 calendar days of delivery."
        ),
        additional_metadata={
            "category": "return_deadline",
            "risk": "high",
        },
    ),
]


northstar_policy_dataset = EvaluationDataset(
    goldens=northstar_policy_goldens
)