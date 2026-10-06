from google import genai
from google.genai import errors
from groq import Groq

from app.config import settings


SYSTEM_INSTRUCTION = """
You are an intelligent support ticket resolution assistant.

Your job is to help resolve customer support tickets using ONLY the
evidence provided to you.

STRICT RULES:

1. Use only the supplied evidence.

2. Do not invent troubleshooting steps, product behavior, policies,
   or technical facts.

3. Historical support tickets are examples of previous cases.
   Their previous resolutions may be useful as supporting precedent,
   but they are NOT authoritative documentation.

4. Authoritative Knowledge Base articles are the primary source
   for troubleshooting procedures and support guidance.

5. Use historical resolutions only when they are consistent with
   the authoritative Knowledge Base evidence.

6. If a historical resolution conflicts with an authoritative
   Knowledge Base article, follow the Knowledge Base article.

7. Every factual resolution step MUST include at least one citation
   such as [S1] or [S2].

8. Citation formatting is mandatory. You MUST write citations exactly
   in the form [S1], [S2], [S3], etc.

9. Every numbered item under "Recommended Action" MUST end with
   at least one citation.

10. Do not produce a "Recommended Action" item without a citation.

11. If the evidence is insufficient, do not invent a solution.
    State that the evidence is insufficient and recommend human
    escalation.

12. Do not fabricate citations.

13. Keep the response practical and concise.

14. If the issue requires information that is not present in the
    evidence, recommend escalation to a human support agent.

15. Historical customer support data may contain outdated or
    incomplete information. Do not treat it as current policy.

16. Preserve normal spacing between all words.

17. Do not concatenate words when generating the response.
    For example, write "device shows", not "deviceshows", and
    "turn it off and on", not "turn it off andon".

18. Use plain text only. Do not use unnecessary Markdown formatting.

19. Leave a blank line between Summary, Recommended Action,
    and Escalation.

20. If the Escalation section contains a factual statement,
    recommendation, or instruction, it MUST end with a citation.
    If no escalation is required, write exactly "Not required".

Return the response in exactly this structure:

Summary:
<brief description of the issue>

Recommended Action:
<numbered steps, with citations>

Escalation:
<"Not required" OR a brief explanation of why human support is needed>
"""


class RAGService:
    def __init__(self):
        if not settings.GEMINI_API_KEY and not settings.GROQ_API_KEY:
            raise ValueError(
                "Neither GEMINI_API_KEY nor GROQ_API_KEY is configured."
            )

        # ---------------------------------------------------------
        # Primary LLM: Gemini
        # ---------------------------------------------------------
        self.gemini_client = None

        if settings.GEMINI_API_KEY:
            self.gemini_client = genai.Client(
                api_key=settings.GEMINI_API_KEY
            )

        # ---------------------------------------------------------
        # Fallback LLM: Groq
        # ---------------------------------------------------------
        self.groq_client = None

        if settings.GROQ_API_KEY:
            self.groq_client = Groq(
                api_key=settings.GROQ_API_KEY
            )

    # =============================================================
    # Evidence construction
    # =============================================================

    def build_evidence(self, retrieval_results):
        """
        Convert retrieval results into clearly separated evidence
        blocks for the LLM.

        Knowledge Base articles are authoritative.
        Historical tickets are supporting examples only.
        """

        evidence_blocks = []

        for index, result in enumerate(
            retrieval_results,
            start=1,
        ):
            citation = f"S{index}"

            source_id = result.get(
                "source_id",
                "unknown",
            )

            source_type = result.get(
                "source_type",
                "unknown",
            )

            title = result.get(
                "title",
                "",
            )

            content = result.get(
                "content",
                "",
            )

            historical_answer = result.get(
                "historical_answer"
            )

            # -----------------------------------------------------
            # Authoritative Knowledge Base
            # -----------------------------------------------------
            if source_type == "knowledge_base":
                block = f"""
[{citation}]
SOURCE TYPE: AUTHORITATIVE KNOWLEDGE BASE
SOURCE ID: {source_id}
TITLE: {title}

CONTENT:
{content}
"""

            # -----------------------------------------------------
            # Historical Support Ticket
            # -----------------------------------------------------
            elif source_type == "historical_ticket":
                block = f"""
[{citation}]
SOURCE TYPE: HISTORICAL SUPPORT TICKET
SOURCE ID: {source_id}
TITLE: {title}

PREVIOUS CUSTOMER ISSUE:
{content}

PREVIOUS HISTORICAL RESOLUTION:
{historical_answer or "No historical resolution available."}

IMPORTANT:
This is historical support information and is NOT authoritative.
Use it only as supporting precedent when consistent with the
authoritative Knowledge Base.
"""

            # -----------------------------------------------------
            # Unknown source type
            # -----------------------------------------------------
            else:
                block = f"""
[{citation}]
SOURCE TYPE: {source_type}
SOURCE ID: {source_id}
TITLE: {title}

CONTENT:
{content}
"""

            evidence_blocks.append(block.strip())

        return "\n\n".join(evidence_blocks)

    # =============================================================
    # Prompt construction
    # =============================================================

    def build_prompt(self, query, retrieval_results):
        evidence = self.build_evidence(
            retrieval_results
        )

        return f"""
{SYSTEM_INSTRUCTION}

CUSTOMER SUPPORT TICKET:
{query}

EVIDENCE:
{evidence}

Now generate the grounded support resolution.

Remember:
- Use ONLY the evidence above.
- Prefer authoritative Knowledge Base information.
- Historical tickets are supporting examples only.
- Every factual recommended action needs a citation.
- Every numbered Recommended Action must have a citation.
- Do not fabricate citations.
- Preserve normal spacing between words.
- Do not concatenate words.
- Follow the required output structure exactly.
- Any factual Escalation statement or recommendation must have a citation.
"""

    # =============================================================
    # Gemini generation
    # =============================================================

    def _generate_with_gemini(self, prompt):
        response = self.gemini_client.models.generate_content(
            model=settings.LLM_MODEL,
            contents=prompt,
        )

        if not response.text:
            raise RuntimeError(
                "Gemini returned an empty response."
            )

        return response.text.strip()

    # =============================================================
    # Groq generation
    # =============================================================

    def _generate_with_groq(self, prompt):
        response = self.groq_client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_INSTRUCTION,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.1,
        )

        answer = response.choices[0].message.content

        if not answer:
            raise RuntimeError(
                "Groq returned an empty response."
            )

        return answer.strip()

    # =============================================================
    # Generate answer
    # =============================================================

    def generate_answer(self, query, retrieval_results):
        """
        Generate a grounded support resolution.

        Gemini is attempted first.
        Groq is used as a fallback if Gemini is unavailable.
        """

        # ---------------------------------------------------------
        # No evidence
        # ---------------------------------------------------------
        if not retrieval_results:
            return {
                "status": "escalate",
                "reason": "no_retrieval_evidence",
                "answer": (
                    "I could not find sufficient support evidence "
                    "to answer this ticket safely."
                ),
                "sources": [],
            }

        # ---------------------------------------------------------
        # Build grounded prompt
        # ---------------------------------------------------------
        prompt = self.build_prompt(
            query,
            retrieval_results,
        )

        # ---------------------------------------------------------
        # Primary LLM: Gemini
        # ---------------------------------------------------------
        if self.gemini_client:
            try:
                answer = self._generate_with_gemini(
                    prompt
                )

                return {
                    "status": "generated",
                    "reason": "gemini_success",
                    "provider": "gemini",
                    "answer": answer,
                    "sources": self._build_sources(
                        retrieval_results
                    ),
                }

            except errors.ServerError:
                print(
                    "Gemini unavailable. Falling back to Groq."
                )

            except errors.ClientError:
                print(
                    "Gemini client error. Falling back to Groq."
                )

            except Exception as exc:
                print(
                    f"Gemini error: {exc}. Falling back to Groq."
                )

        # ---------------------------------------------------------
        # Fallback LLM: Groq
        # ---------------------------------------------------------
        if self.groq_client:
            try:
                answer = self._generate_with_groq(
                    prompt
                )

                return {
                    "status": "generated",
                    "reason": "groq_fallback_success",
                    "provider": "groq",
                    "answer": answer,
                    "sources": self._build_sources(
                        retrieval_results
                    ),
                }

            except Exception as exc:
                return {
                    "status": "escalate",
                    "reason": "all_llms_failed",
                    "answer": (
                        "The support assistant could not generate "
                        "a reliable response. Please route this "
                        "ticket to a human support agent."
                    ),
                    "sources": self._build_sources(
                        retrieval_results
                    ),
                    "error": str(exc),
                }

        # ---------------------------------------------------------
        # No LLM available
        # ---------------------------------------------------------
        return {
            "status": "escalate",
            "reason": "no_llm_available",
            "answer": (
                "The support assistant could not generate a response. "
                "Please route this ticket to a human support agent."
            ),
            "sources": self._build_sources(
                retrieval_results
            ),
        }

    # =============================================================
    # Source metadata
    # =============================================================

    def _build_sources(self, retrieval_results):
        """
        Build source metadata returned by the API.
        """

        return [
            {
                "citation": f"S{index}",
                "source_id": result.get(
                    "source_id",
                    "unknown",
                ),
                "title": result.get(
                    "title"
                ),
                "rrf_score": result.get(
                    "rrf_score"
                ),
            }
            for index, result in enumerate(
                retrieval_results,
                start=1,
            )
        ]


rag_service = RAGService()