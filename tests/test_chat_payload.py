"""Contract tests for the chat payload schema."""
from __future__ import annotations

from typing import Any, Dict

import pytest

from chatbot import ChatPipeline, SafetyAdvisor, SafetyAdvisorConfig


@pytest.fixture()
def pipeline_with_trace_enabled() -> ChatPipeline:
    """Pipeline with detector trace exposure forced on."""

    safety_config = SafetyAdvisorConfig(expose_detector_trace=True)
    safety_advisor = SafetyAdvisor(config=safety_config)
    return ChatPipeline(safety_advisor=safety_advisor)


@pytest.fixture()
def pipeline_with_trace_disabled() -> ChatPipeline:
    """Default pipeline leaving detector trace off."""

    return ChatPipeline()


def assert_base_payload_structure(response: Dict[str, Any]) -> None:
    """Ensure base payload fields exist with expected types."""

    assert set(response.keys()) == {
        "reply",
        "emotions",
        "risk_level",
        "safety_actions",
        "retrieved_context",
        "metadata",
        "safety",
        "coaching",
    }
    assert isinstance(response["reply"], str)
    assert isinstance(response["emotions"], list)
    assert isinstance(response["risk_level"], str)
    assert isinstance(response["safety_actions"], list)
    assert isinstance(response["retrieved_context"], list)

    metadata = response["metadata"]
    assert isinstance(metadata, dict)
    assert "emotion_confidence" in metadata
    assert "risk_confidence" in metadata

    safety = response["safety"]
    assert isinstance(safety, dict)
    assert set(safety.keys()) == {"disclaimer", "guidance", "escalation_contacts"}
    assert isinstance(response["coaching"], list)


def test_chat_payload_without_detector_trace(pipeline_with_trace_disabled: ChatPipeline) -> None:
    """Default configuration excludes detector trace metadata."""

    response = pipeline_with_trace_disabled.run_chat(
        user_id="user-123",
        user_message="I am feeling okay today",
        modalities=["text"],
    )

    assert_base_payload_structure(response)
    assert "detector_trace" not in response["metadata"]


def test_chat_payload_with_detector_trace(pipeline_with_trace_enabled: ChatPipeline) -> None:
    """Feature flag exposes detector trace when enabled."""

    response = pipeline_with_trace_enabled.run_chat(
        user_id="user-456",
        user_message="I am feeling really frustrated",
        modalities=["text"],
    )

    assert_base_payload_structure(response)
    metadata = response["metadata"]
    assert "detector_trace" in metadata
    assert isinstance(metadata["detector_trace"], dict)


def test_confidence_driven_escalation_actions(pipeline_with_trace_disabled: ChatPipeline) -> None:
    """High confidence risk and emotion detections trigger extra support."""

    response = pipeline_with_trace_disabled.run_chat(
        user_id="user-789",
        user_message="I am overwhelmed and have been thinking about suicide",
        modalities=["text"],
    )

    assert_base_payload_structure(response)
    assert set(response["safety_actions"]) >= {
        "escalate_confidence_based",
        "reinforce_emotion_support",
        "suggest_immediate_help",
        "list_crisis_hotline",
        "append_disclaimer",
    }

    guidance_text = "\n".join(response["safety"]["guidance"])
    assert "strong signs of distress" in guidance_text
    assert "feelings of crisis" in guidance_text