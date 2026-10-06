from app.llm import llm_service


class QueryAnalyzer:
    """
    LLM-based customer support ticket analyzer.

    Gemini is the primary LLM provider.
    Groq is used automatically as a fallback.

    The LLM determines:
    - intent
    - product
    - severity
    - sentiment
    - normalized_query
    - confidence
    """

    SYSTEM_PROMPT = """
You are the query-analysis component of an intelligent telecom
customer support resolution system.

Analyze the customer's support complaint and return ONLY a valid
JSON object.

Your job is to understand the complaint semantically. Do NOT use
keyword matching rules.

Determine:

1. intent
   - Identify the primary support problem.
   - Use a concise snake_case label.
   - Examples:
     mobile_data_not_working
     slow_mobile_data
     no_network_signal
     call_drop
     poor_call_quality
     sms_not_working
     sim_issue
     esim_issue
     roaming_issue
     apn_issue
     account_access
     billing_issue
     recharge_issue
     plan_issue
     data_allowance_exhausted
     network_outage
     network_degradation
     connectivity_issue
     general_troubleshooting
   - If the complaint represents a new class not covered by these
     examples, create an appropriate concise snake_case intent.
   - Do not force the complaint into an incorrect existing class.

2. product
   Identify the telecom product/service involved.

   Examples:
   mobile_data
   mobile_network
   voice_calls
   sms
   sim
   esim
   roaming
   account
   billing
   recharge
   plan
   broadband
   fixed_line
   unknown

   If the complaint clearly concerns a product not in the examples,
   provide an appropriate concise snake_case product.

3. severity
   Choose exactly one:
   low
   medium
   high

   Consider:
   - service impact
   - duration/repetition
   - inability to access an important service
   - business/work impact
   - urgency
   - security/fraud implications
   - customer impact described in the complaint

   Do not determine severity from a single keyword alone.

4. sentiment
   Determine the customer's actual emotional tone from the entire
   complaint.

   Choose exactly one:
   positive
   neutral
   frustrated
   angry
   dissatisfied
   urgent

   Sentiment MUST be inferred semantically from the complaint.
   Do not use keyword-based sentiment rules.

5. normalized_query
   Produce a concise semantic representation of the actual problem.
   Preserve important details such as the affected service,
   recurring behavior, or relevant context.

6. confidence
   Give your confidence in the analysis as a number from 0.0 to 1.0.

Important:
- Return JSON only.
- Do not wrap the JSON in markdown.
- Do not explain your reasoning.
- Do not invent customer facts.
- Do not confuse sentiment with severity.
- A frustrated customer is not necessarily a high-severity customer.
"""

    def analyze(self, ticket: str) -> dict:
        """
        Analyze a customer support ticket using the LLM service.

        Gemini is attempted first and Groq is automatically used if
        Gemini fails.
        """

        if not ticket or not ticket.strip():
            return {
                "intent": "unknown",
                "product": "unknown",
                "severity": "low",
                "sentiment": "neutral",
                "normalized_query": "",
                "confidence": 0.0,
                "provider": "none",
            }

        prompt = f"""
{self.SYSTEM_PROMPT}

Customer support complaint:

{ticket.strip()}
"""

        try:
            result = llm_service.generate_json(prompt)

            provider = result.pop("_provider", "unknown")

            analysis = {
                "intent": self._clean_value(
                    result.get("intent"),
                    "unknown",
                ),
                "product": self._clean_value(
                    result.get("product"),
                    "unknown",
                ),
                "severity": self._normalize_severity(
                    result.get("severity")
                ),
                "sentiment": self._normalize_sentiment(
                    result.get("sentiment")
                ),
                "normalized_query": self._clean_value(
                    result.get("normalized_query"),
                    ticket.strip(),
                ),
                "confidence": self._normalize_confidence(
                    result.get("confidence")
                ),
                "provider": provider,
            }

            return analysis

        except Exception as exc:
            print(
                f"[ANALYZER] LLM analysis failed: {exc}"
            )

            return {
                "intent": "unknown",
                "product": "unknown",
                "severity": "low",
                "sentiment": "neutral",
                "normalized_query": ticket.strip(),
                "confidence": 0.0,
                "provider": "none",
            }

    @staticmethod
    def _clean_value(value, default: str) -> str:
        """
        Safely normalize an LLM string field.
        """

        if value is None:
            return default

        value = str(value).strip()

        if not value:
            return default

        return value

    @staticmethod
    def _normalize_severity(value) -> str:
        """
        Normalize severity to the supported API values.
        """

        if not value:
            return "low"

        value = str(value).strip().lower()

        if value in {"low", "medium", "high"}:
            return value

        return "low"

    @staticmethod
    def _normalize_sentiment(value) -> str:
        """
        Normalize sentiment without performing sentiment analysis.

        The classification itself comes entirely from the LLM.
        This method only validates the returned value.
        """

        if not value:
            return "neutral"

        value = str(value).strip().lower()

        allowed = {
            "positive",
            "neutral",
            "frustrated",
            "angry",
            "dissatisfied",
            "urgent",
        }

        if value in allowed:
            return value

        # Normalize a broader LLM sentiment label.
        # This does not perform sentiment analysis; the LLM
        # already determined the sentiment.
        if value == "negative":
            return "dissatisfied"

        return "neutral"

    @staticmethod
    def _normalize_confidence(value) -> float:
        """
        Safely normalize the LLM confidence score.
        """

        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return 0.0

        return max(0.0, min(1.0, confidence))


query_analyzer = QueryAnalyzer()
