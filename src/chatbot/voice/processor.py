"""Voice processing utilities with optional Whisper transcription support."""
from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency hook
    import whisper  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency hook
    whisper = None  # type: ignore[assignment]


@dataclass
class VoiceConfig:
    """Configuration flags that control voice processing behaviour."""

    enable_voice_transcription: bool = True
    enable_voice_emotion_detection: bool = True
    whisper_model: str = "base"


class VoiceProcessor:
    """Convert audio payloads to transcripts and coarse emotion hints."""

    def __init__(self, config: VoiceConfig | None = None) -> None:
        self.config = config or VoiceConfig()
        self._model = None

    def process_audio(self, audio_base64: str) -> Tuple[str, List[str]]:
        """Decode audio input and produce a transcript plus emotion hints."""

        audio_data = self._decode_audio_to_array(audio_base64)

        transcript = ""
        model = self._ensure_model()
        if model is not None and self.config.enable_voice_transcription:
            try:
                result = model.transcribe(audio_data)
                transcript = result.get("text", "").strip()
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Whisper transcription failed; continuing without transcript: %s", exc)
                transcript = ""

        voice_emotions = self._detect_voice_emotions(transcript, audio_data)

        return transcript, voice_emotions

    def extract_audio_metadata(self, audio_base64: str) -> Dict[str, float | str]:
        """Return lightweight diagnostics about the audio payload."""

        audio_data = self._decode_audio_to_array(audio_base64)
        duration_seconds = len(audio_data) / 16000.0  # Assuming 16kHz sample rate

        return {
            "duration_seconds": round(duration_seconds, 2),
            "encoding": "wav",
        }

    def _ensure_model(self):
        """Load Whisper on demand, gracefully disabling it when unavailable."""

        if not self.config.enable_voice_transcription:
            return None

        if whisper is None:
            logger.warning("`openai-whisper` is not installed; disabling voice transcription.")
            self.config.enable_voice_transcription = False
            return None

        if self._model is None:
            try:
                self._model = whisper.load_model(self.config.whisper_model)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning(
                    "Failed to load Whisper model '%s'; disabling voice transcription: %s",
                    self.config.whisper_model,
                    exc,
                )
                self.config.enable_voice_transcription = False
                self._model = None
        return self._model

    def _decode_audio_to_array(self, audio_base64: str) -> np.ndarray:
        """Decode base64 audio to a mono numpy array."""
        try:
            audio_bytes = base64.b64decode(audio_base64)
            audio_data, _ = sf.read(io.BytesIO(audio_bytes))
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)
            return audio_data.astype(np.float32)
        except Exception as exc:  # pragma: no cover - validation guard
            raise ValueError("Invalid base64-encoded audio input") from exc

    def _detect_voice_emotions(self, transcript: str, audio_data: np.ndarray) -> List[str]:
        """Apply simple heuristics to infer voice emotion cues."""

        if not self.config.enable_voice_emotion_detection:
            return []

        keywords = (
            ("anger", ("angry", "mad", "furious", "irritated")),
            ("sadness", ("sad", "cry", "tearful", "depressed")),
            ("joy", ("happy", "excited", "relieved", "grateful")),
        )

        lower_transcript = (transcript or "").lower()
        for label, synonyms in keywords:
            if any(word in lower_transcript for word in synonyms):
                return [label]

        # Fall back to a crude energy-based heuristic when no keywords are present.
        energy = float(np.mean(np.abs(audio_data))) if len(audio_data) else 0.0
        if energy > 0.25:
            return ["anger"]
        if energy < 0.05:
            return ["sadness"]

        return ["neutral"]