"""Vision-based emotion detector using FER and OpenCV."""
from __future__ import annotations

import base64
from functools import lru_cache
from typing import Dict, Optional

import cv2
import numpy as np
from fer import FER


@lru_cache(maxsize=1)
def _build_fer_detector() -> FER:
    """Create the FER detector with MTCNN face detection enabled."""

    return FER(mtcnn=True)


def _decode_base64_image(image_b64: str) -> np.ndarray:
    """Convert a base64-encoded image string into an OpenCV BGR frame."""

    if image_b64.startswith("data:"):
        _, encoded = image_b64.split(",", maxsplit=1)
    else:
        encoded = image_b64
    buffer = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
    frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Unable to decode image data")
    return frame


def detect_emotions_from_base64(image_b64: str) -> Dict[str, float]:
    """Infer emotions from a base64-encoded frame.

    Returns a dictionary mapping emotion labels to confidence scores sorted by
    descending probability.
    """

    frame = _decode_base64_image(image_b64)
    return detect_emotions_from_frame(frame)


def detect_emotions_from_frame(frame: np.ndarray) -> Dict[str, float]:
    """Infer emotions directly from an OpenCV frame."""

    detector = _build_fer_detector()
    predictions = detector.detect_emotions(frame)
    if not predictions:
        return {}

    emotions = predictions[0]["emotions"]
    sorted_items = sorted(emotions.items(), key=lambda item: item[1], reverse=True)
    return dict(sorted_items)


class VisionEmotionDetector:
    """Convenience wrapper maintaining FER state."""

    def __init__(self) -> None:
        self._detector = _build_fer_detector()

    def detect_from_base64(self, image_b64: str) -> Dict[str, float]:
        frame = _decode_base64_image(image_b64)
        return self.detect_from_frame(frame)

    def detect_from_frame(self, frame: np.ndarray) -> Dict[str, float]:
        predictions = self._detector.detect_emotions(frame)
        if not predictions:
            return {}
        emotions = predictions[0]["emotions"]
        sorted_items = sorted(emotions.items(), key=lambda item: item[1], reverse=True)
        return dict(sorted_items)