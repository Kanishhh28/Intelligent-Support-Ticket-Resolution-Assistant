from app.analyzer import query_analyzer
from app.retrieval import hybrid_search
from app.rag import rag_service
from app.validation import validation_service


class SupportResolutionPipeline:
    """
    End-to-end support ticket resolution pipeline.

    Flow:

        Customer Ticket
              ↓
        Query Analyzer
              ↓
        Hybrid Retrieval
              ↓
        RRF
              ↓
        Metadata Soft Boost
              ↓
        RAG Generation
              ↓
        Validation
              ↓
        Answer / Escalate
    """

    def __init__(self):
        self.analyzer = query_analyzer
        self.rag_service = rag_service
        self.validation_service = validation_service

    def resolve(
        self,
        query: str,
        retrieval_limit: int = 5,
    ) -> dict:

        # ---------------------------------------------------------
        # 1. VALIDATE INPUT
        # ---------------------------------------------------------

        if not query or not query.strip():
            return {
                "status": "escalate",
                "reason": "empty_query",
                "answer": (
                    "The support ticket is empty. "
                    "Please provide the customer's issue."
                ),
                "analysis": {
                    "intent": "unknown",
                    "product": "unknown",
                    "severity": "low",
                    "sentiment": "neutral",
                },
                "sources": [],
            }

        query = query.strip()

        # ---------------------------------------------------------
        # DEBUG PIPELINE HEADER
        # ---------------------------------------------------------

        print("\n")
        print("=" * 60)
        print("           SUPPORT RESOLUTION DEBUG PIPELINE")
        print("=" * 60)

        # ---------------------------------------------------------
        # 1. CUSTOMER QUERY
        # ---------------------------------------------------------

        print("\n[1] CUSTOMER QUERY")
        print(f"    {query}")

        # ---------------------------------------------------------
        # 2. ANALYZE
        # ---------------------------------------------------------

        analysis = self.analyzer.analyze(query)

        print("\n[2] QUERY ANALYSIS")
        print(
            f"    Intent     : "
            f"{analysis.get('intent', '—')}"
        )
        print(
            f"    Product    : "
            f"{analysis.get('product', '—')}"
        )
        print(
            f"    Severity   : "
            f"{analysis.get('severity', '—')}"
        )
        print(
            f"    Sentiment  : "
            f"{analysis.get('sentiment', '—')}"
        )
        print(
            f"    Confidence : "
            f"{analysis.get('confidence', '—')}"
        )

        # ---------------------------------------------------------
        # 2B. UNSUPPORTED / UNKNOWN QUERY GATE
        # ---------------------------------------------------------
        #
        # Do not generate an answer when the analyzer cannot map
        # the ticket to a supported product or intent.
        #
        # This prevents historical tickets from being used as
        # authoritative troubleshooting evidence for unsupported
        # classes such as broadband.
        #

        if (
            analysis.get("intent") == "unknown"
            or analysis.get("product") == "unknown"
        ):
            print("\n[FINAL RESULT]")
            print("=" * 60)
            print("    Status : ESCALATE")
            print("    Reason : unsupported_or_unknown_class")
            print("=" * 60 + "\n")

            return {
                "status": "escalate",
                "reason": "unsupported_or_unknown_class",
                "answer": (
                    "The ticket could not be mapped to a supported "
                    "support category with sufficient confidence. "
                    "Please route this ticket to a human support agent."
                ),
                "analysis": analysis,
                "sources": [],
            }

        # ---------------------------------------------------------
        # 3. RETRIEVE
        # ---------------------------------------------------------
        #
        # The analyzer output is passed into hybrid_search so that
        # product, intent, and severity can be used as soft ranking
        # signals.
        #
        # We do NOT hard-filter documents based on these fields.
        #

        retrieval_results = hybrid_search(
            query=query,
            limit=retrieval_limit,
            analysis=analysis,
        )

        print("\n[3] RETRIEVAL")
        print(
            f"    Sources retrieved : "
            f"{len(retrieval_results)}"
        )

        for index, source in enumerate(
            retrieval_results,
            start=1,
        ):
            print(
                f"    S{index} : "
                f"{source.get('title', 'Unknown source')}"
            )

        # ---------------------------------------------------------
        # 4. CHECK RETRIEVAL
        # ---------------------------------------------------------

        if not retrieval_results:
            print("\n[FINAL RESULT]")
            print("=" * 60)
            print("    Status : ESCALATE")
            print("    Reason : no_retrieval_evidence")
            print("=" * 60 + "\n")

            return {
                "status": "escalate",
                "reason": "no_retrieval_evidence",
                "answer": (
                    "I could not find sufficient support evidence "
                    "to resolve this ticket safely."
                ),
                "analysis": analysis,
                "sources": [],
            }

        # ---------------------------------------------------------
        # 5. GENERATE
        # ---------------------------------------------------------

        rag_result = self.rag_service.generate_answer(
            query=query,
            retrieval_results=retrieval_results,
        )

        print("\n[4] RAG / LLM GENERATION")

        provider = rag_result.get("provider")

        if provider == "gemini":
            print("    Gemini : SUCCESS")
            print("    Groq   : NOT USED")

        elif provider == "groq":
            print("    Gemini : FAILED")
            print("    Groq   : SUCCESS")
            print("    Mode   : Gemini fallback")

        else:
            print(
                f"    Provider : "
                f"{provider or 'NONE'}"
            )

        # ---------------------------------------------------------
        # 5. RAG ANSWER
        # ---------------------------------------------------------

        print("\n[5] RAG ANSWER")
        print("-" * 60)
        print(
            rag_result.get(
                "answer",
                "No answer generated.",
            )
        )
        print("-" * 60)

        # ---------------------------------------------------------
        # 6. LLM FAILURE → ESCALATE
        # ---------------------------------------------------------

        if rag_result["status"] == "escalate":

            print("\n[FINAL RESULT]")
            print("=" * 60)
            print("    Status : ESCALATE")
            print(
                f"    Reason : "
                f"{rag_result.get('reason', 'unknown')}"
            )
            print("=" * 60 + "\n")

            return {
                "status": "escalate",
                "reason": rag_result["reason"],
                "answer": rag_result["answer"],
                "analysis": analysis,
                "sources": rag_result.get(
                    "sources",
                    [],
                ),
            }

        # ---------------------------------------------------------
        # 7. VALIDATE
        # ---------------------------------------------------------

        validation_result = self.validation_service.validate(
            answer=rag_result["answer"],
            retrieval_results=retrieval_results,
        )

        print("\n[6] VALIDATION")
        print(
            f"    Status : "
            f"{validation_result.get('status', '—')}"
        )
        print(
            f"    Reason : "
            f"{validation_result.get('reason', '—')}"
        )

        if "validation_score" in validation_result:
            print(
                f"    Score  : "
                f"{validation_result.get('validation_score')}"
            )

        # ---------------------------------------------------------
        # 8. ANSWER
        # ---------------------------------------------------------

        if validation_result["status"] == "answer":

            print("\n" + "=" * 60)
            print("FINAL RESULT")
            print("=" * 60)
            print("    Status   : ANSWER")
            print(
                f"    Provider : "
                f"{rag_result.get('provider', 'unknown').upper()}"
            )
            print("=" * 60 + "\n")

            return {
                "status": "answer",
                "reason": "validation_passed",
                "answer": rag_result["answer"],
                "analysis": analysis,
                "sources": rag_result["sources"],
                "validation": validation_result,
            }

        # ---------------------------------------------------------
        # 9. VALIDATION FAILURE → ESCALATE
        # ---------------------------------------------------------

        print("\n" + "=" * 60)
        print("FINAL RESULT")
        print("=" * 60)
        print("    Status : ESCALATE")
        print(
            f"    Reason : "
            f"{validation_result.get('reason', 'unknown')}"
        )
        print("=" * 60 + "\n")

        return {
            "status": "escalate",
            "reason": validation_result["reason"],
            "answer": (
                "The available evidence was not sufficient to "
                "produce a safely validated resolution. "
                "Please route this ticket to a human support agent."
            ),
            "analysis": analysis,
            "sources": rag_result["sources"],
            "validation": validation_result,
        }


support_pipeline = SupportResolutionPipeline()