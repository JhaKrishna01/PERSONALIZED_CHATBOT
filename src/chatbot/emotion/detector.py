"""Emotion detector using transformers for text classification."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence

try:
    from transformers import pipeline
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pipeline = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

VOICE_FALLBACK_CONFIDENCE = 0.5
DEFAULT_NEUTRAL_CONFIDENCE = 0.5

_FALLBACK_LABELS: Sequence[str] = (
    "sadness",
    "joy",
    "anger",
    "fear",
    "neutral",
    "love",
    "surprise",
    "disgust",
    "crisis",
)

_KEYWORD_EMOTION_MAP: Dict[str, Sequence[str]] = {
    "joy": (
        "happy",
        "relieved",
        "grateful",
        "excited",
        "optimistic",
        "good",
    ),
    "sadness": (
        "sad",
        "down",
        "depressed",
        "hopeless",
        "cry",
        "tearful",
        "overwhelmed",
        "lonely",
    ),
    "anger": (
        "angry",
        "mad",
        "furious",
        "frustrated",
        "annoyed",
        "irritated",
    ),
    "fear": (
        "anxious",
        "worried",
        "scared",
        "afraid",
        "panic",
        "terrified",
        "nervous",
    ),
    "surprise": (
        "surprised",
        "shocked",
        "unexpected",
        "astonished",
    ),
    "disgust": (
        "disgusted",
        "gross",
        "nasty",
    ),
    "love": (
        "love",
        "caring",
        "supportive",
        "affection",
    ),
    "crisis": (
        "suicide",
        "self-harm",
        "self harm",
        "end my life",
        "kill myself",
        "can't go on",
        "harm myself",
    ),
}


class _KeywordEmotionEstimator:
    """Heuristic fallback when the transformers pipeline is unavailable."""

    def __call__(self, transcript: str) -> List[List[Dict[str, float]]]:
        text = (transcript or "").lower()
        scores: Dict[str, float] = {label: 0.05 for label in _FALLBACK_LABELS}
        matched = False

        for label, keywords in _KEYWORD_EMOTION_MAP.items():
            if any(keyword in text for keyword in keywords):
                scores[label] = max(scores[label], 0.9)
                matched = True

        if not matched:
            scores["neutral"] = max(scores.get("neutral", 0.0), 0.82)
        else:
            scores["neutral"] = min(scores.get("neutral", 0.2), 0.2)

        return [[{"label": label, "score": float(scores[label])} for label in _FALLBACK_LABELS]]


@lru_cache(maxsize=1)
def _get_emotion_classifier():
    """Load the preferred emotion classifier with graceful fallbacks."""

    if pipeline is None:
        logger.info("`transformers` is not installed; using keyword-based emotion estimation.")
        return _KeywordEmotionEstimator()

    try:
        return pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True,
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning(
            "Falling back to keyword-based emotion estimator because the transformer pipeline could not be loaded: %s",
            exc,
        )
        return _KeywordEmotionEstimator()


@dataclass
class EmotionDetectionResult:
    """Structured result from the emotion detector."""

    emotions: List[str]
    risk_level: str
    confidence: Dict[str, float]
    risk_confidence: float
    model_trace: Dict[str, Any] | None = None


class EmotionDetector:
    """Emotion detector using Hugging Face transformers with keyword fallback."""

    def __init__(self) -> None:
        self.classifier = _get_emotion_classifier()

    def detect(
        self,
        *,
        transcript: str,
        voice_emotions: Optional[List[str]] = None,
    ) -> EmotionDetectionResult:
        # Use model (or fallback) for emotion classification
        results = self.classifier(transcript)
        ranked_scores: Sequence[Dict[str, float]] = []

        if isinstance(results, list) and results:
            first = results[0]
            if isinstance(first, list):
                ranked_scores = first
            elif isinstance(first, dict):  # pragma: no cover - alternate pipeline format
                ranked_scores = results  # type: ignore[assignment]

        scores: Dict[str, float] = {
            item["label"]: float(item["score"])
            for item in ranked_scores
            if "label" in item and "score" in item
        }

        if not scores:
            scores = {"neutral": DEFAULT_NEUTRAL_CONFIDENCE}

        threshold = 0.3
        detected_emotions = [label for label, score in scores.items() if score >= threshold]
        confidence = dict(scores)

        # Merge voice-derived emotions if provided
        if voice_emotions:
            for emotion in voice_emotions:
                if emotion not in detected_emotions:
                    detected_emotions.append(emotion)
                voice_confidence = max(confidence.get(emotion, 0.0), VOICE_FALLBACK_CONFIDENCE)
                confidence[emotion] = min(max(voice_confidence, 0.0), 1.0)

        if not detected_emotions:
            detected_emotions.append("neutral")
            confidence["neutral"] = max(confidence.get("neutral", 0.0), DEFAULT_NEUTRAL_CONFIDENCE)

        # Deduplicate while preserving order
        seen = set()
        ordered_emotions = []
        for emotion in detected_emotions:
            if emotion not in seen:
                ordered_emotions.append(emotion)
                seen.add(emotion)
        detected_emotions = ordered_emotions

        # Determine risk level based on emotions
        crisis_keywords = ["fear", "anger", "sadness", "crisis"]
        high_risk = any(confidence.get(emotion, 0.0) > 0.7 for emotion in crisis_keywords)

        max_confidence = max(confidence.values(), default=DEFAULT_NEUTRAL_CONFIDENCE)
        if high_risk or "crisis" in detected_emotions:
            risk_level = "high"
            risk_confidence = max(0.75, min(0.95, max_confidence))
        elif len(detected_emotions) > 1:
            risk_level = "moderate"
            risk_confidence = max_confidence * 0.8
        else:
            risk_level = "low"
            risk_confidence = max_confidence * 0.6

        model_trace: Dict[str, Any] = {
            "model_scores": confidence,
            "voice_emotions": voice_emotions or [],
            "estimator": type(self.classifier).__name__,
        }

        return EmotionDetectionResult(
            emotions=detected_emotions,
            risk_level=risk_level,
            confidence=confidence,
            risk_confidence=round(min(max(risk_confidence, 0.0), 1.0), 3),