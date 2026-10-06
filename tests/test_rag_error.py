from app.rag import rag_service


def test_rag_without_evidence():

    result = rag_service.generate_answer(
        query="My product is not working.",
        retrieval_results=[],
    )

    print("\nNO EVIDENCE TEST")
    print(result)

    assert result["status"] == "escalate"
    assert result["reason"] == "no_retrieval_evidence"


if __name__ == "__main__":

    test_rag_without_evidence()

    print("\nRAG error-handling test passed.")