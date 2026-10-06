from app.rag import rag_service


def main():

    query = (
        "The product is performing poorly "
        "and I need help improving its efficiency."
    )

    retrieval_results = [
        {
            "source_id": "TICKET-000010",
            "title": "Problem with Product Efficiency",
            "content": (
                "A customer reported that the product "
                "was performing poorly and requested "
                "assistance with product efficiency."
            ),
            "intent": "Problem",
            "severity": "high",
            "rrf_score": 0.01639,
        },
        {
            "source_id": "TICKET-000545",
            "title": "",
            "content": (
                "A previous customer reported a "
                "similar product performance issue."
            ),
            "intent": "Problem",
            "severity": "medium",
            "rrf_score": 0.01612,
        },
    ]

    result = rag_service.generate_answer(
        query=query,
        retrieval_results=retrieval_results,
    )

    print("\nRAG FALLBACK RESULT")
    print("Status:", result["status"])
    print("Reason:", result["reason"])
    print("\nAnswer:")
    print(result["answer"])
    print("\nSources:")
    print(result["sources"])

    assert result["status"] == "generated"
    assert result["reason"] in [
        "gemini_success",
        "groq_fallback_success",
    ]
    assert result["answer"]
    assert len(result["sources"]) == 2

    print("\nRAG fallback test passed.")


if __name__ == "__main__":
    main()