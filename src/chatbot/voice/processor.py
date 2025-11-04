"""Voice processing using Whisper for transcription."""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import whisper


@dataclass
class VoiceConfig:
    """Configuration for voice processing behavior."""

    enable_voice_transcription: bool = True
    enable_voice_emotion_detection: bool = True
    whisper_model: str = "base"


class VoiceProcessor:
    """Convert audio to text using Whisper and extract emotion cues."""

    def __init__(self, config: VoiceConfig | None = None) -> None:
        self.config = config or VoiceConfig()
        self.model = whisper.load_model(self.config.whisper_model) if self.config.enable_voice_transcription else None

    def process_audio(self, audio_base64: str) -> Tuple[str, List[str]]:
        """Decode audio input and produce a transcript plus emotion hints."""

        audio_data = self._decode_audio_to_array(audio_base64)

        transcript = ""
        if self.config.enable_voice_transcription and self.model:
            result = self.model.transcribe(audio_data)
            transcript = result["text"].strip()

        voice_emotions = self._detect_voice_emotions(transcript, audio_data)

        return transcript, voice_emotions

    def extract_audio_metadata(self, audio_base64: str) -> Dict[str, float | str]:
        """Return lightweight diagnostics about the audio payload."""

        audio_data = self._decode_audio_to_array(audio_base64)

        duration_seconds = len(audio_data) / 16000.0  # Assuming 16kHz

        return {
            "duration_seconds": round(duration_seconds, 2),
            "encoding": "wav",
        }

    def _decode_audio_to_array(self, audio_base64: str) -> np.ndarray:
        """Decode base64 audio to numpy array."""
        try:
            audio_bytes = base64.b64decode(audio_base64)
            # Assume it's WAV format
            audio_data, _ = sf.read(io.BytesIO(audio_bytes))
            # Convert to mono if stereo
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)
            return audio_data
        except Exception as exc:
            raise ValueError("Invalid base64-encoded audio input") from exc

    def _detect_voice_emotions(self, transcript: str, audio_data: np.ndarray) -> List[str]:
        if not self.config.enable_voice_emotion_detection:
            return []

        # Simple heuristic: check for keywords in transcript or analyze pitch
        if transcript:
            lower_transcript = transcript.lower()
            if "angry" in lower_transcript or "mad" in lower_transcript:
                return ["anger"]
            elif "sad" in lower_transcript or "cry" in lower_transcript:
                return ["sadness"]
            elif "happy" in lower_transcript or "excited" in lower_transcript:
                return ["joy"]

        # Basic pitch analysis (rough estimate)
        # This is very simplistic; in real implementation, use proper voice emotion detection
        # For now, just return "neutral"
        return ["neutral"]