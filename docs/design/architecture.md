# System Architecture Overview

## 1. High-Level Flow
1. **Client Input**: User sends text and optionally audio via `/api/v1/chat`.
2. **Voice Processing** (optional): Audio bytes decoded, transcribed, and enriched with paralinguistic cues by `VoiceProcessor`.
3. **Emotion & Risk Detection**: `EmotionDetector` analyzes text (and voice cues) to produce emotion labels, confidence scores, and a risk assessment.
4. **Personalized Retrieval**: `VectorRetrievalStub` fetches user history, coping strategies, and psychoeducational material. Replace with true vector DB integration.
5. **Therapeutic Response Generation**: `TherapeuticResponder` synthesizes the user message, emotional cues, and retrieved snippets to craft a response based on therapeutic strategies (e.g., NURSE).
6. **Safety Layer**: `SafetyAdvisor` injects disclaimers and escalates to crisis protocols when necessary.
7. **API Response**: Aggregated data returned to the client, including generated reply, detected emotions, risk level, safety actions, and contextual snippets.

## 2. Module Responsibilities
- **`chatbot/routes/v1`**: Flask blueprint exposing `health` and `chat` endpoints. Responsible for request validation and response formatting.
- **`chatbot/services/chat_pipeline.py`**: Orchestrator that wires together emotion detection, vector retrieval, LLM responses, and safety checks.
- **`chatbot/emotion`**: Houses the emotion detection logic. Replace heuristics with transformer-based classifiers and risk-scoring models.
- **`chatbot/vector_stub`**: Placeholder for vector database access. Keep the method signature stable for easy substitution with real Pinecone/Weaviate/Chroma implementations.
- **`chatbot/llm`**: Generates therapeutic responses. Intended to integrate with managed LLM APIs or fine-tuned local models.
- **`chatbot/safety`**: Applies crisis escalation rules, disclaimers, and regulatory guardrails.
- **`chatbot/voice`**: Entry point for speech-to-text and voice emotion analysis.

## 3. Future Enhancements
- Replace heuristic emotion detection with multi-label classifiers (e.g., BERT, RoBERTa) and fine-tuned crisis detection models.
- Integrate speech pipelines (Whisper, wav2vec) for accurate transcription and pitch/tempo-based emotion insights.
- Swap `VectorRetrievalStub` with production-grade vector databases. Include caching, user-specific namespaces, and PII-safe storage.
- Implement prompt templates, response moderation, and conversation memory for the LLM layer.
- Expand `SafetyAdvisor` to trigger actual helpline APIs, compliance logging, and human-on-call notifications.

## 4. Deployment Considerations
- Containerize the Flask app and deploy behind HTTPS with authentication.
- Secure environment variables for LLM keys, vector database credentials, and third-party APIs.
- Add observability (structured logging, metrics) for monitoring safety triggers and performance.

## 5. Testing Strategy
- **Unit Tests**: Validate emotion detector heuristics, pipeline glue logic, and safety rules with controlled inputs.
- **Integration Tests**: Simulate end-to-end chat requests covering text-only and text+audio modalities.
- **Safety Regression Tests**: Ensure crisis language always leads to the correct escalation actions.
- **Voice Tests**: Validate audio decoding and error handling when adding real speech-to-text modules.