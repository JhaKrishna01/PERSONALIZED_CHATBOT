"""API v1 endpoints for the chatbot."""
from __future__ import annotations

from flask import Blueprint, jsonify, request

from ...services.chat_pipeline import ChatPipeline


chat_pipeline = ChatPipeline()


def register_endpoints(bp: Blueprint) -> None:
    """Register API routes on the provided blueprint."""

    @bp.get("/health")
    def health_check():
        """Simple health check endpoint."""
        return jsonify({"status": "ok"})

    @bp.post("/chat")
    def chat():
        """Primary chat endpoint accepting user input and optional metadata."""
        payload = request.get_json(force=True, silent=True) or {}
        user_id = payload.get("user_id", "anonymous")
        user_input = payload.get("message")
        modalities = payload.get("modalities", ["text"])  # e.g., ["text"], ["text", "audio"]

        if not user_input:
            return (
                jsonify({"error": "Request requires non-empty 'message'."}),
                400,
            )

        response_payload = chat_pipeline.run_chat(
            user_id=user_id,
            user_message=user_input,
            modalities=modalities,
            audio_bytes=payload.get("audio_base64"),
            face_image_b64=payload.get("face_image_b64"),
        )

        return jsonify(response_payload)

    @bp.get("/history/<string:user_id>")
    def history(user_id: str):
        """Expose recent conversation history for demo purposes."""
        from ...services.db import fetch_history

        turns = fetch_history(user_id=user_id, max_turns=20)
        serialized = [
            {
                "id": turn.id,
                "user_message": turn.user_message,
                "bot_reply": turn.bot_reply,
                "emotions": turn.emotions.split(",") if turn.emotions else [],
                "created_at": turn.created_at.isoformat() if turn.created_at else None,
            }
            for turn in turns
        ]
        return jsonify({"user_id": user_id, "turns": serialized})