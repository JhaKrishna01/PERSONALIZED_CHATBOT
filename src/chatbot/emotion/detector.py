"""Emotion detector using transformers for text classification."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional

from transformers import pipeline

VOICE_FALLBACK_CONFIDENCE = 0.5
DEFAULT_NEUTRAL_CONFIDENCE = 0.5


@lru_cache(maxsize=1)
def _get_emotion_classifier():
    """Load the emotion classification pipeline."""
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)


@dataclass
class EmotionDetectionResult:
    """Structured result from the emotion detector."""

    emotions: List[str]
    risk_level: str
    confidence: Dict[str, float]
    risk_confidence: float
    model_trace: Dict[str, Any] | None = None


class EmotionDetector:
    """Emotion detector using Hugging Face transformers."""

    def __init__(self):
        self.classifier = _get_emotion_classifier()

    def detect(
        self,
        transcript: str,
        voice_emotions: Optional[List[str]] = None,
    ) -> EmotionDetectionResult:
        # Use model for emotion classification
        results = self.classifier(transcript)

        # results is a list of dicts with scores for each label
        scores = {item['label']: item['score'] for item in results[0]}

        # Filter emotions above a threshold, say 0.3
        threshold = 0.3
        detected_emotions = [label for label, score in scores.items() if score >= threshold]

        confidence = scores

        # Merge voice-derived emotions if provided
        if voice_emotions:
            for emotion in voice_emotions:
                detected_emotions.append(emotion)
                voice_confidence = max(confidence.get(emotion, 0.0), VOICE_FALLBACK_CONFIDENCE)
                confidence[emotion] = voice_confidence

        if not detected_emotions:
            detected_emotions.append("neutral")
            confidence["neutral"] = DEFAULT_NEUTRAL_CONFIDENCE

        # Determine risk level based on emotions
        crisis_keywords = ["fear", "anger", "sadness"]  # High risk if strong negative emotions
        high_risk = any(confidence.get(emotion, 0) > 0.7 for emotion in crisis_keywords)

        risk_level: str
        risk_confidence: float
        if high_risk or "crisis" in detected_emotions:
            risk_level = "high"
            risk_confidence = 0.9
        elif len(detected_emotions) > 1:
            risk_level = "moderate"
            risk_confidence = max(confidence.values()) * 0.8
        else:
            risk_level = "low"
            risk_confidence = max(confidence.values()) * 0.6

        model_trace: Dict[str, Any] = {
            "model_scores": scores,
            "voice_emotions": voice_emotions or [],
        }

        return EmotionDetectionResult(
            emotions=detected_emotions,
            risk_level=risk_level,
            confidence=confidence,
            risk_confidence=round(min(risk_confidence, 1.0), 3),
            model_trace=model_trace,
        )