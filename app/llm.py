import json
from typing import Any

from google import genai
from groq import Groq

from app.config import settings


class LLMService:
    """
    Centralized LLM service.

    Primary provider:
        Gemini

    Fallback provider:
        Groq

    All application components should use this service instead of
    calling Gemini or Groq directly.
    """

    def __init__(self):
        self.gemini_client = None
        self.groq_client = None

        if settings.GEMINI_API_KEY:
            self.gemini_client = genai.Client(
                api_key=settings.GEMINI_API_KEY
            )

        if settings.GROQ_API_KEY:
            self.groq_client = Groq(
                api_key=settings.GROQ_API_KEY
            )

        self.gemini_model = settings.LLM_MODEL
        self.groq_model = settings.GROQ_MODEL

    # =========================================================
    # GEMINI
    # =========================================================

    def _call_gemini(self, prompt: str) -> str:
        if not self.gemini_client:
            raise RuntimeError("Gemini client is not configured.")

        response = self.gemini_client.models.generate_content(
            model=self.gemini_model,
            contents=prompt,
        )

        text = getattr(response, "text", None)

        if not text:
            raise RuntimeError("Gemini returned an empty response.")

        return text.strip()

    # =========================================================
    # GROQ FALLBACK
    # =========================================================

    def _call_groq(self, prompt: str) -> str:
        if not self.groq_client:
            raise RuntimeError("Groq client is not configured.")

        response = self.groq_client.chat.completions.create(
            model=self.groq_model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0,
        )

        text = response.choices[0].message.content

        if not text:
            raise RuntimeError("Groq returned an empty response.")

        return text.strip()

    # =========================================================
    # GEMINI → GROQ FALLBACK
    # =========================================================

    def generate(self, prompt: str) -> dict[str, Any]:
        """
        Generate an LLM response.

        Gemini is always attempted first.
        Groq is used only if Gemini fails.

        Returns:
            {
                "provider": "gemini" | "groq",
                "text": "...",
            }
        """

        gemini_error = None

        try:
            text = self._call_gemini(prompt)

            return {
                "provider": "gemini",
                "text": text,
            }

        except Exception as exc:
            gemini_error = str(exc)

            print(
                f"[LLM] Gemini failed. "
                f"Falling back to Groq: {gemini_error}"
            )

        try:
            text = self._call_groq(prompt)

            return {
                "provider": "groq",
                "text": text,
            }

        except Exception as groq_error:
            raise RuntimeError(
                "Both Gemini and Groq failed. "
                f"Gemini error: {gemini_error}; "
                f"Groq error: {groq_error}"
            ) from groq_error

    # =========================================================
    # JSON GENERATION
    # =========================================================

    def generate_json(self, prompt: str) -> dict[str, Any]:
        """
        Generate a JSON response using Gemini first and Groq
        as fallback.

        The model is instructed to return JSON only.
        """

        response = self.generate(prompt)

        text = response["text"].strip()

        # Remove accidental markdown fences.
        if text.startswith("```"):
            lines = text.splitlines()

            if lines and lines[0].startswith("```"):
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            text = "\n".join(lines).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # LLMs can occasionally emit a minor trailing-quote
            # formatting error such as:
            # {"confidence":0.95"}
            # Retry only this narrowly scoped repair.
            repaired_text = text

            if repaired_text.endswith('"}'):
                repaired_text = repaired_text[:-2] + '}'
            elif repaired_text.endswith('"]'):
                repaired_text = repaired_text[:-2] + ']'

            try:
                data = json.loads(repaired_text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"LLM returned invalid JSON from "
                    f"{response['provider']}: {text}"
                ) from exc

        if not isinstance(data, dict):
            raise ValueError(
                "LLM JSON response must be an object."
            )

        data["_provider"] = response["provider"]

        return data


llm_service = LLMService()
