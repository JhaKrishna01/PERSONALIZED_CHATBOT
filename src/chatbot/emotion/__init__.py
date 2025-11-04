"""Emotion and risk detection package."""

from .detector import EmotionDetector, EmotionDetectionResult  # noqa: F401
# from .vision_detector import VisionEmotionDetector  # noqa: F401

__all__ = [
    "EmotionDetector",
    "EmotionDetectionResult",
    # "VisionEmotionDetector",
]