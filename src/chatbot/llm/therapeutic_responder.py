"""Therapeutic responder backed by Google Gemini."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from ..config import get_env

try:
    from google.api_core import exceptions as google_exceptions
except ModuleNotFoundError:  # pragma: no cover - dependency guard
    google_exceptions = None  # type: ignore[assignment]

from .client import get_generative_model


logger = logging.getLogger(__name__)

SUPPORTIVE_FALLBACK_MESSAGE = (
    "I'm having trouble accessing my support tools right now, but I'm still here with you. "
    "Let's take a calming breath together and focus on one supportive step you can take next. "
    "If you're in immediate danger, please reach out to emergency services or someone you trust."
)
NO_HISTORY_MESSAGE = "No previous conversation history recorded."


@dataclass
class TherapeuticResponderConfig:
    """Configuration for the therapeutic responder."""

    model_name: str = "gemini-1.5-pro"
    safety_instructions: str = (
        "You are a compassionate mental health support companion following the "
        "NURSE framework (Name emotion, Understand, Respect, Support, Explore). "
        "Provide short, empathetic responses, encourage coping strategies, and "
        "remind the user to seek professional help when appropriate."
    )
    max_output_tokens: int = 350


class TherapeuticResponder:
    """Generate empathetic responses using Google Gemini."""

    def __init__(self, config: TherapeuticResponderConfig | None = None) -> None:
        self.config = config or TherapeuticResponderConfig()
        # Allow overriding via environment if needed.
        self.model_name, self.api_version = self._resolve_model_and_version()

    def _resolve_model_and_version(self) -> Tuple[str, str | None]:
        env_model = get_env("GEMINI_MODEL_NAME", default=self.config.model_name)
        env_version = get_env("GEMINI_API_VERSION", default=None)

        if "@" in env_model:
            # Split explicit version suffix (e.g., "models/gemini-1.5-pro-latest@001").
            name, version = env_model.split("@", maxsplit=1)
            return name, version

        return env_model, env_version

    def _build_prompt(
        self,
        user_message: str,
        emotions: Sequence[str],
        risk_level: str,
        retrieved_context: Sequence[str],
        conversation_history: Optional[Sequence[str]],
    ) -> List[str]:
        context_section = "\n".join(retrieved_context) if retrieved_context else "No personalized history retrieved."
        emotion_summary = ", ".join(emotions) if emotions else "neutral"
        history_section = "\n".join(conversation_history) if conversation_history else NO_HISTORY_MESSAGE

        return [
            self.config.safety_instructions,
            "Recent conversation history:\n" + history_section,
            "Relevant context: \n" + context_section,
            f"Detected emotional cues: {emotion_summary}",
            f"Assessed risk level: {risk_level}",
            "User message:",
            user_message,
            "Compose a supportive reply addressing the emotions, respecting the risk level, and offering next-step suggestions.",
        ]

    def _handle_generation_error(self, exc: Exception) -> str:
        """Map Gemini exceptions to a supportive fallback message."""

        if google_exceptions:
            if isinstance(exc, google_exceptions.NotFound):
                logger.error("Gemini model '%s' not found: %s", self.model_name, exc)
                logger.info("Falling back to supportive static response due to missing model")
                return SUPPORTIVE_FALLBACK_MESSAGE
            if isinstance(exc, google_exceptions.InvalidArgument):
                logger.error("Invalid request to Gemini model '%s': %s", self.model_name, exc)
                logger.info("Falling back to supportive static response due to invalid configuration")
                return SUPPORTIVE_FALLBACK_MESSAGE
            if isinstance(exc, google_exceptions.GoogleAPIError):
                logger.exception("Unexpected Gemini API error")
                logger.info("Falling back to supportive static response due to API error: %s", exc)
                return SUPPORTIVE_FALLBACK_MESSAGE

        logger.exception("Unhandled error while generating therapeutic response")
        logger.info("Falling back to supportive static response due to unexpected error: %s", exc)
        return SUPPORTIVE_FALLBACK_MESSAGE

    def generate_response(
        self,
        user_message: str,
        emotions: Sequence[str],
        risk_level: str,
        retrieved_context: Sequence[str],
        conversation_history: Optional[Sequence[str]] = None,
    ) -> str:
        model = get_generative_model(self.model_name)
        prompt_parts = self._build_prompt(
            user_message,
            emotions,
            risk_level,
            retrieved_context,
            conversation_history,
        )

        try:
            generation_config = {
                "temperature": 0.85,
                "top_p": 0.9,
                "top_k": 40,
                "max_output_tokens": self.config.max_output_tokens,
            }

            # Pass api_version when provided (required for some fine-grained SKUs).
            if self.api_version:
                generation_config["api_version"] = self.api_version

            response = model.generate_content(
                prompt_parts,
                generation_config=generation_config,
            )
        except Exception as exc:  # pragma: no cover - defensive logging only
            return self._handle_generation_error(exc)

        candidate_text: str | None = None
        if response:
            try:
                if getattr(response, "text", None):
                    candidate_text = response.text
                elif getattr(response, "candidates", None):
                    for candidate in response.candidates:
                        if getattr(candidate, "content", None):
                            parts = getattr(candidate.content, "parts", None)
                            if parts:
                                # Concatenate all text parts while ignoring non-text parts.
                                texts = [getattr(part, "text", "") for part in parts if getattr(part, "text", None)]
                                candidate_text = "\n".join(filter(None, texts))
                                if candidate_text:
                                    break
                # As a final fallback, inspect the first part directly if available.
                if not candidate_text and getattr(response, "candidates", None):
                    first_candidate = response.candidates[0]
                    first_part = getattr(getattr(first_candidate, "content", None), "parts", [None])[0]
                    candidate_text = getattr(first_part, "text", None)
            except Exception as parse_exc:  # pragma: no cover - defensive logging only
                logger.warning("Unable to parse Gemini response structure: %s", parse_exc)

        if not candidate_text:
            if getattr(logger, "warning", None):
                logger.warning(
                    "Gemini response returned no usable text. finish_reason=%s",
                    getattr(response.candidates[0], "finish_reason", "unknown") if getattr(response, "candidates", None) else "missing",
                )
            return SUPPORTIVE_FALLBACK_MESSAGE

        return candidate_text.strip()