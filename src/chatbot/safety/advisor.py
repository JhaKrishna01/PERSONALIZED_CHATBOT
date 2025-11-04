"""Safety layer that injects disclaimers and escalation guidance."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from ..config import get_bool_env, get_float_env


@dataclass
class SafetyOutcome:
    """Result produced by the safety advisor."""

    actions: List[str] = field(default_factory=list)
    disclaimer: str | None = None
    guidance_messages: List[str] = field(default_factory=list)
    escalation_contacts: List[str] = field(default_factory=list)


@dataclass
class SafetyAdvisorConfig:
    """Configuration toggles for the safety advisor."""

    expose_detector_trace: bool = False
    risk_confidence_threshold: float = 0.75
    emotion_confidence_threshold: float = 0.75


class SafetyAdvisor:
    """Placeholder safety evaluator.

    Replace with policies that integrate clinical oversight, helpline escalation,
    and content moderation models.
    """

    def __init__(self, config: SafetyAdvisorConfig | None = None) -> None:
        if config is None:
            config = SafetyAdvisorConfig(
                expose_detector_trace=get_bool_env("EXPOSE_DETECTOR_TRACE", default=False),
                risk_confidence_threshold=get_float_env("SAFETY_RISK_THRESHOLD", 0.75),
                emotion_confidence_threshold=get_float_env("SAFETY_EMOTION_THRESHOLD", 0.75),
            )
        self.config = config

    def evaluate(
        self,
        message: str,
        llm_reply: str,
        risk_level: str,
        *,
        risk_confidence: float,
        emotion_confidence: dict[str, float],
    ) -> SafetyOutcome:
        outcome = SafetyOutcome(actions=[])

        # Universal disclaimer and baseline guidance
        outcome.actions.append("append_disclaimer")
        outcome.disclaimer = (
            "I’m not a medical professional, but I can help you find resources."
        )
        outcome.guidance_messages.append(
            "If you feel overwhelmed, consider reaching out to a trusted person or a professional for additional support."
        )

        if self.config.expose_detector_trace:
            outcome.actions.append("include_detector_trace")

        message_lower = message.lower()
        crisis_terms = (
            "suicide",
            "self-harm",
            "self harm",
            "end my life",
            "kill myself",
            "can't go on",
            "cant go on",
            "harm myself",
        )
        crisis_language_detected = any(term in message_lower for term in crisis_terms)

        if risk_confidence >= self.config.risk_confidence_threshold or crisis_language_detected:
            if "escalate_confidence_based" not in outcome.actions:
                outcome.actions.append("escalate_confidence_based")
            outcome.guidance_messages.append(
                "Because we detected strong signs of distress, consider reaching out to a crisis support resource or trusted contact immediately."
            )

        high_confidence_emotions = [
            label
            for label, score in emotion_confidence.items()
            if score >= self.config.emotion_confidence_threshold
        ]
        if high_confidence_emotions:
            outcome.actions.append("reinforce_emotion_support")
            summarized_emotions = ", ".join(sorted(set(high_confidence_emotions)))
            outcome.guidance_messages.append(
                f"We're here with you while you navigate feelings of {summarized_emotions}. Try naming what you need right now and reach for grounding tools that have helped before."
            )

        if risk_level == "high":
            outcome.actions.extend([
                "suggest_immediate_help",
                "list_crisis_hotline",
            ])
            outcome.guidance_messages.extend([
                "If you are in immediate danger, contact local emergency services right away.",
                "You deserve support—please connect with someone you trust or a trained counselor as soon as possible.",
            ])
            outcome.escalation_contacts.append(
                "Find a helpline in your region: https://findahelpline.com/"
            )
        elif risk_level == "moderate":
            outcome.guidance_messages.append(
                "Consider practicing a grounding exercise (deep breathing, journaling) and then check in with someone you trust."
            )

        return outcome