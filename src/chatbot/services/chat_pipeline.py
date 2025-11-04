"""Core orchestration pipeline for the chatbot."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from ..emotion import EmotionDetector  # , VisionEmotionDetector
from ..llm import TherapeuticResponder
from ..safety import SafetyAdvisor
from ..services.db import ConversationTurn, fetch_history, save_turn
from ..vector_stub import VectorRetrievalStub
from ..voice import VoiceProcessor


@dataclass
class PipelineResult:
    """Structured output of the chat pipeline."""

    reply: str
    detected_emotions: List[str]
    risk_level: str
    safety_actions: List[str]
    retrieved_context: List[str]
    metadata: Dict[str, Any]


class ChatPipeline:
    """Glue code that brings together emotion detection, retrieval, LLM, and safety layers."""

    def __init__(
        self,
        emotion_detector: Optional[EmotionDetector] = None,
        vector_retrieval: Optional[VectorRetrievalStub] = None,
        responder: Optional[TherapeuticResponder] = None,
        safety_advisor: Optional[SafetyAdvisor] = None,
        voice_processor: Optional[VoiceProcessor] = None,
        # vision_detector: Optional[VisionEmotionDetector] = None,
    ) -> None:
        self.emotion_detector = emotion_detector or EmotionDetector()
        self.vector_retrieval = vector_retrieval or VectorRetrievalStub()
        self.responder = responder or TherapeuticResponder()
        self.safety_advisor = safety_advisor or SafetyAdvisor()
        self.voice_processor = voice_processor or VoiceProcessor()
        # self.vision_detector = vision_detector or VisionEmotionDetector()

    def run_chat(
        self,
        user_id: str,
        user_message: str,
        modalities: List[str],
        audio_bytes: Optional[str] = None,
        face_image_b64: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute the end-to-end chat flow."""
        transcript = user_message
        voice_emotions: List[str] = []

        if "audio" in modalities and audio_bytes:
            transcript, voice_emotions = self.voice_processor.process_audio(audio_bytes)

        vision_emotions: Dict[str, float] = {}
        # if face_image_b64:
        #     try:
        #         vision_emotions = self.vision_detector.detect_from_base64(face_image_b64)
        #     except Exception as exc:
        #         vision_emotions = {"error": str(exc)}

        consolidated_emotions = list(voice_emotions)
        consolidated_emotions.extend(list(vision_emotions.keys())[:1])

        emotion_result = self.emotion_detector.detect(
            transcript=transcript,
            voice_emotions=consolidated_emotions or None,
        )

        retrieved_snippets = self.vector_retrieval.fetch_personalized_context(
            user_id=user_id,
            query=transcript,
        )

        history_turns: Iterable[ConversationTurn] = fetch_history(user_id=user_id, max_turns=10)
        formatted_history = [
            f"User: {turn.user_message}\nAssistant: {turn.bot_reply}"
            for turn in reversed(list(history_turns))
        ]

        llm_reply = self.responder.generate_response(
            user_message=transcript,
            emotions=emotion_result.emotions,
            risk_level=emotion_result.risk_level,
            retrieved_context=retrieved_snippets,
            conversation_history=formatted_history,
        )

        safety_outcome = self.safety_advisor.evaluate(
            message=transcript,
            llm_reply=llm_reply,
            risk_level=emotion_result.risk_level,
            risk_confidence=emotion_result.risk_confidence,
            emotion_confidence=emotion_result.confidence,
        )

        advisor_config = getattr(self.safety_advisor, "config", {})
        expose_detector_trace = getattr(advisor_config, "expose_detector_trace", None)
        should_include_trace = bool(expose_detector_trace)
        metadata: Dict[str, Any] = {
            "user_id": user_id,
            "modality": modalities,
            "emotion_confidence": emotion_result.confidence,
            "risk_confidence": emotion_result.risk_confidence,
            # "vision_emotions": vision_emotions,
        }

        if should_include_trace and emotion_result.model_trace is not None:
            metadata["detector_trace"] = emotion_result.model_trace

        result = PipelineResult(
            reply=llm_reply,
            detected_emotions=emotion_result.emotions,
            risk_level=emotion_result.risk_level,
            safety_actions=safety_outcome.actions,
            retrieved_context=retrieved_snippets,
            metadata=metadata,
        )

        save_turn(
            user_id=user_id,
            message=transcript,
            reply=llm_reply,
            emotions=emotion_result.emotions,
        )

        # Add personalized context to vector DB
        self.vector_retrieval.add_user_context(user_id, f"User said: {transcript}")
        self.vector_retrieval.add_user_context(user_id, f"Assistant replied: {llm_reply}")

        response = {
            "reply": result.reply,
            "emotions": result.detected_emotions,
            "risk_level": result.risk_level,
            "safety_actions": result.safety_actions,
            "retrieved_context": result.retrieved_context,
            "metadata": result.metadata,
            "safety": {
                "disclaimer": safety_outcome.disclaimer or "",
                "guidance": safety_outcome.guidance_messages,
                "escalation_contacts": safety_outcome.escalation_contacts,
            },
            "coaching": safety_outcome.guidance_messages[:],
        }

        return response