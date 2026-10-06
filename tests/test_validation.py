from app.validation import validation_service


retrieved_results = [
    {
        "source_id": "KB-001",
        "title": "Product Troubleshooting Guide",
        "content": "Restart the product and verify the configuration.",
    },
    {
        "source_id": "TICKET-000010",
        "title": "Problem with Product Efficiency",
        "content": "A previous customer reported product efficiency problems.",
    },
    {
        "source_id": "TICKET-000020",
        "title": "Product Configuration Issue",
        "content": "A previous ticket involved incorrect configuration.",
    },
]


def test_valid_answer():
    answer = """
Summary:
The customer is experiencing a product efficiency problem.

Recommended Action:
1. Restart the product and verify the configuration. [S1]
2. Check whether the issue continues after the configuration is verified. [S1]

Escalation:
Not required
"""

    result = validation_service.validate(
        answer,
        retrieved_results,
    )

    print("\nVALID ANSWER TEST")
    print(result)

    assert result["status"] == "answer"
    assert result["valid"] is True
    assert result["reason"] == "validation_passed"


def test_missing_citations():
    answer = """
Summary:
The customer is experiencing a product efficiency problem.

Recommended Action:
1. Restart the product.
2. Check the configuration.

Escalation:
Not required
"""

    result = validation_service.validate(
        answer,
        retrieved_results,
    )

    print("\nMISSING CITATIONS TEST")
    print(result)

    assert result["status"] == "escalate"
    assert result["reason"] == "missing_citations"


def test_invalid_citation():
    answer = """
Summary:
The customer is experiencing a product efficiency problem.

Recommended Action:
1. Restart the product. [S99]

Escalation:
Not required
"""

    result = validation_service.validate(
        answer,
        retrieved_results,
    )

    print("\nINVALID CITATION TEST")
    print(result)

    assert result["status"] == "escalate"
    assert result["reason"] == "invalid_citations"
    assert "S99" in result["invalid_citations"]


def test_uncited_recommendation():
    answer = """
Summary:
The customer is experiencing a product efficiency problem.

Recommended Action:
1. Restart the product. [S1]
2. Reinstall the entire application.

Escalation:
Not required
"""

    result = validation_service.validate(
        answer,
        retrieved_results,
    )

    print("\nUNCITED RECOMMENDATION TEST")
    print(result)

    assert result["status"] == "escalate"
    assert result["reason"] == "uncited_recommendation"
    assert result["missing_steps"] == [2]


def test_explicit_insufficient_evidence():
    answer = """
Summary:
The customer is experiencing a product efficiency problem.

Recommended Action:
The available evidence is insufficient to safely resolve the issue. [S1]

Escalation:
Escalation to a human support agent is required because
the retrieved evidence does not contain actionable
troubleshooting procedures.
"""

    result = validation_service.validate(
        answer,
        retrieved_results,
    )

    print("\nEXPLICIT INSUFFICIENT EVIDENCE TEST")
    print(result)

    assert result["status"] == "escalate"
    assert result["reason"] == "insufficient_evidence"


def test_escalation_not_required():
    answer = """
Summary:
The customer is experiencing a product efficiency problem.

Recommended Action:
1. Restart the product and verify the configuration. [S1]

Escalation:
Not required
"""

    result = validation_service.validate(
        answer,
        retrieved_results,
    )

    print("\nESCALATION NOT REQUIRED TEST")
    print(result)

    assert result["status"] == "answer"
    assert result["valid"] is True
    assert result["reason"] == "validation_passed"


if __name__ == "__main__":

    test_valid_answer()
    test_missing_citations()
    test_invalid_citation()
    test_uncited_recommendation()
    test_explicit_insufficient_evidence()
    test_escalation_not_required()

    print("\nAll validation tests passed.")