import re


CITATION_PATTERN = re.compile(r"\[S(\d+)\]")


class ValidationService:
    """
    Deterministic validation gate for generated support answers.

    Checks:
    1. The answer is not empty.
    2. Evidence citations exist.
    3. Citations refer to retrieved evidence.
    4. Recommended action steps contain citations.
    5. Explicit escalation or insufficient-evidence responses
       are routed to a human agent.

    This does not perform full semantic fact-checking.
    """

    def validate(
        self,
        answer: str,
        retrieval_results: list[dict],
    ) -> dict:

        # ---------------------------------------------------------
        # 1. Basic checks
        # ---------------------------------------------------------

        if not answer or not answer.strip():
            return self._escalate(
                reason="empty_answer",
                details="The LLM returned an empty answer.",
            )

        if not retrieval_results:
            return self._escalate(
                reason="no_retrieval_evidence",
                details="No evidence was retrieved for the ticket.",
            )

        # ---------------------------------------------------------
        # 2. Detect explicit escalation
        # ---------------------------------------------------------

        if self._requires_escalation(answer):
            return self._escalate(
                reason="insufficient_evidence",
                details=(
                    "The generated response determined that the "
                    "available evidence is insufficient for a safe "
                    "automated resolution."
                ),
            )

        # ---------------------------------------------------------
        # 3. Extract citations
        # ---------------------------------------------------------

        cited_numbers = [
            int(number)
            for number in CITATION_PATTERN.findall(answer)
        ]

        cited_numbers = sorted(set(cited_numbers))

        # Valid citation IDs correspond to retrieved results:
        #
        # First result  -> S1
        # Second result -> S2
        # etc.

        valid_citation_numbers = set(
            range(1, len(retrieval_results) + 1)
        )

        invalid_citations = [
            number
            for number in cited_numbers
            if number not in valid_citation_numbers
        ]

        # ---------------------------------------------------------
        # 4. Check that citations exist
        # ---------------------------------------------------------

        if not cited_numbers:
            return self._escalate(
                reason="missing_citations",
                details=(
                    "The generated answer contains no "
                    "evidence citations."
                ),
            )

        # ---------------------------------------------------------
        # 5. Check for fabricated citations
        # ---------------------------------------------------------

        if invalid_citations:
            return self._escalate(
                reason="invalid_citations",
                details=(
                    "The answer contains citations that do not "
                    "correspond to retrieved evidence: "
                    f"{invalid_citations}"
                ),
                citations=cited_numbers,
                invalid_citations=invalid_citations,
            )

        # ---------------------------------------------------------
        # 6. Check Recommended Action section
        # ---------------------------------------------------------

        action_section = self._extract_action_section(answer)

        if action_section:

            missing_step_citations = self._find_uncited_steps(
                action_section
            )

            if missing_step_citations:
                return self._escalate(
                    reason="uncited_recommendation",
                    details=(
                        "One or more recommended action steps "
                        "do not contain an evidence citation."
                    ),
                    citations=cited_numbers,
                    missing_steps=missing_step_citations,
                )

        # ---------------------------------------------------------
        # 7. Validation passed
        # ---------------------------------------------------------

        citation_coverage = min(
            len(cited_numbers) / len(retrieval_results),
            1.0,
        )

        validation_score = round(
            0.5 + (0.5 * citation_coverage),
            3,
        )

        return {
            "status": "answer",
            "valid": True,
            "reason": "validation_passed",
            "validation_score": validation_score,
            "citations": [
                f"S{number}"
                for number in cited_numbers
            ],
            "invalid_citations": [],
            "missing_steps": [],
        }

    # =============================================================
    # Detect explicit escalation
    # =============================================================

    def _requires_escalation(
        self,
        answer: str,
    ) -> bool:
        """
        Detect whether the generated response explicitly requires
        immediate human escalation or states that the evidence is
        insufficient.

        Conditional escalation guidance is allowed. For example:

            If the issue persists after troubleshooting, escalate
            to a support agent.

        should not cause the entire response to be rejected.
        """

        answer_lower = answer.lower()

        # ---------------------------------------------------------
        # Check the Escalation section specifically
        # ---------------------------------------------------------

        escalation_match = re.search(
            r"escalation\s*:\s*(.*?)(?:\n\s*$|\Z)",
            answer_lower,
            re.DOTALL,
        )

        if escalation_match:

            escalation_text = escalation_match.group(1).strip()

            # Successful automated resolution
            if escalation_text == "not required":
                return False

            # -----------------------------------------------------
            # Conditional escalation is valid guidance, not an
            # immediate escalation decision.
            # -----------------------------------------------------

            conditional_patterns = [
                r"\bif\b",
                r"\bwhen\b",
                r"\bunless\b",
                r"\bonly if\b",
                r"\bshould .* persist\b",
                r"\bcontinues? to\b",
                r"\bafter .* troubleshooting\b",
                r"\bafter .* steps\b",
                r"\bafter .* checks\b",
                r"\bif .* remains?\b",
            ]

            is_conditional = any(
                re.search(pattern, escalation_text)
                for pattern in conditional_patterns
            )

            if is_conditional:
                return False

            # -----------------------------------------------------
            # Explicit immediate escalation
            # -----------------------------------------------------

            escalation_patterns = [
                r"human\s+support\s+agent",
                r"human\s+support",
                r"support\s+agent\s+is\s+required",
                r"support\s+agent\s+should\s+review",
                r"requires\s+human\s+support",
                r"requires\s+escalation",
                r"escalation\s+required",
                r"should\s+be\s+escalated",
                r"must\s+be\s+escalated",
            ]

            if any(
                re.search(pattern, escalation_text)
                for pattern in escalation_patterns
            ):
                return True

        # ---------------------------------------------------------
        # Check for insufficient evidence anywhere in answer
        # ---------------------------------------------------------

        evidence_patterns = [
            r"insufficient\s+evidence",
            r"evidence\s+is\s+insufficient",
            r"available\s+evidence\s+is\s+insufficient",
            r"available\s+evidence.*not\s+sufficient",
            r"evidence.*not\s+sufficient",
            r"cannot\s+be\s+safely\s+resolved",
            r"cannot\s+safely\s+resolve",
            r"not\s+enough\s+evidence",
            r"not\s+enough\s+information",
            r"insufficient\s+information",
        ]

        return any(
            re.search(
                pattern,
                answer_lower,
            )
            for pattern in evidence_patterns
        )


    # =============================================================
    # Extract Recommended Action section
    # =============================================================

    def _extract_action_section(
        self,
        answer: str,
    ) -> str:
        """
        Extract the text between:

            Recommended Action:

        and:

            Escalation:

        """

        pattern = re.compile(
            r"Recommended Action\s*:\s*"
            r"(.*?)"
            r"(?:\n\s*Escalation\s*:|$)",
            re.IGNORECASE | re.DOTALL,
        )

        match = pattern.search(answer)

        if not match:
            return ""

        return match.group(1).strip()

    # =============================================================
    # Find uncited numbered steps
    # =============================================================

    def _find_uncited_steps(
        self,
        action_section: str,
    ) -> list[int]:
        """
        Find numbered recommendation steps that contain
        no evidence citation.
        """

        steps = re.findall(
            r"(?m)^\s*(\d+)\.\s*(.+?)(?=^\s*\d+\.|\Z)",
            action_section,
            re.DOTALL,
        )

        missing = []

        for number, step_text in steps:

            if not CITATION_PATTERN.search(step_text):
                missing.append(int(number))

        return missing

    # =============================================================
    # Escalation response
    # =============================================================

    def _escalate(
        self,
        reason: str,
        details: str,
        citations: list[int] | None = None,
        invalid_citations: list[int] | None = None,
        missing_steps: list[int] | None = None,
    ) -> dict:

        return {
            "status": "escalate",
            "valid": False,
            "reason": reason,
            "details": details,
            "validation_score": 0.0,
            "citations": [
                f"S{number}"
                for number in (citations or [])
            ],
            "invalid_citations": [
                f"S{number}"
                for number in (invalid_citations or [])
            ],
            "missing_steps": missing_steps or [],
        }


validation_service = ValidationService()