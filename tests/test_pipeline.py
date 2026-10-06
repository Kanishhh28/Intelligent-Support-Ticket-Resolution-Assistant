from app.pipeline import SupportResolutionPipeline


class FakeAnalyzer:
    """
    Fake analyzer used only for testing the pipeline.

    This avoids calling the real LLM during pipeline tests.
    """

    def analyze(self, query):
        return {
            "intent": "general_troubleshooting",
            "product": "mobile_data",
            "severity": "medium",
            "sentiment": "frustrated",
            "normalized_query": query,
            "confidence": 1.0,
            "provider": "test",
        }


class FakeRAGService:
    """
    Fake RAG service used only for testing the pipeline.

    This avoids calling Gemini or Groq during the test.
    """

    def __init__(self, answer):
        self.answer = answer

    def generate_answer(self, query, retrieval_results):
        return {
            "status": "generated",
            "reason": "llm_success",
            "answer": self.answer,
            "sources": [
                {
                    "citation": f"S{i}",
                    "source_id": result["source_id"],
                    "title": result.get("title"),
                    "rrf_score": result.get("rrf_score"),
                }
                for i, result in enumerate(
                    retrieval_results,
                    start=1,
                )
            ],
        }


class FakeValidationService:
    """
    Fake validation service.

    The real validator will be tested separately.
    """

    def __init__(self, result):
        self.result = result

    def validate(self, answer, retrieval_results):
        return self.result


def test_pipeline_answer_path():

    pipeline = SupportResolutionPipeline()
    pipeline.analyzer = FakeAnalyzer()

    pipeline.rag_service = FakeRAGService(
        """
Summary:
The product is experiencing an efficiency problem.

Recommended Action:
1. Review the product configuration. [S1]

Escalation:
Not required
"""
    )

    pipeline.validation_service = FakeValidationService(
        {
            "status": "answer",
            "valid": True,
            "reason": "validation_passed",
            "validation_score": 1.0,
            "citations": ["S1"],
            "invalid_citations": [],
            "missing_steps": [],
        }
    )

    result = pipeline.resolve(
        "The product is performing poorly and I need help improving its efficiency."
    )

    print("\nANSWER PATH")
    print(result)

    assert result["status"] == "answer"
    assert result["reason"] == "validation_passed"
    assert "answer" in result
    assert "sources" in result
    assert "validation" in result


def test_pipeline_escalation_path():

    pipeline = SupportResolutionPipeline()
    pipeline.analyzer = FakeAnalyzer()

    pipeline.rag_service = FakeRAGService(
        """
Summary:
The product is experiencing an efficiency problem.

Recommended Action:
1. Perform an unsupported action.

Escalation:
Not required
"""
    )

    pipeline.validation_service = FakeValidationService(
        {
            "status": "escalate",
            "valid": False,
            "reason": "uncited_recommendation",
            "validation_score": 0.0,
            "citations": [],
            "invalid_citations": [],
            "missing_steps": [1],
        }
    )

    result = pipeline.resolve(
        "The product is performing poorly and I need help improving its efficiency."
    )

    print("\nESCALATION PATH")
    print(result)

    assert result["status"] == "escalate"
    assert result["reason"] == "uncited_recommendation"
    assert "human support agent" in result["answer"]


def test_empty_query():

    pipeline = SupportResolutionPipeline()

    result = pipeline.resolve("")

    print("\nEMPTY QUERY PATH")
    print(result)

    assert result["status"] == "escalate"
    assert result["reason"] == "empty_query"


if __name__ == "__main__":

    test_pipeline_answer_path()
    test_pipeline_escalation_path()
    test_empty_query()

    print("\nAll pipeline tests passed.")