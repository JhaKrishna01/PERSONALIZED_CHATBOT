# Personalized, Emotion-Aware Conversational Assistant

Prototype implementation of a mental-health-oriented chatbot that blends emotion detection, retrieval-augmented responses, safety guardrails, and optional voice processing.

## Features
- Flask-based API with `/api/v1/chat` endpoint.
- Centralized configuration helpers (`get_env`, `get_bool_env`, `get_float_env`) with environment-driven safety toggles.
- Safety advisor that applies disclaimers, confidence-based escalation, and exposes detector traces when enabled.
- Emotion detection stub tailored for text today with hooks for future voice-informed signals.
- Voice pipeline stub featuring `VoiceConfig` toggles, audio metadata extraction, and placeholders for transcription and emotion analysis.
- Therapeutic response generator following the NURSE framework mindset.
- Vector retrieval stub that returns personalized context snippets for downstream LLM responses.

## Project Layout
```
PERSONALIZED_CHATBOT/
├── README.md
├── pyproject.toml
├── docs/
│   └── design/
│       └── architecture.md
├── src/
│   └── chatbot/
│       ├── __init__.py
│       ├── app.py
│       ├── wsgi.py
│       ├── routes/
│       │   └── v1/
│       │       ├── __init__.py
│       │       └── endpoints.py
│       ├── services/
│       │   └── chat_pipeline.py
│       ├── emotion/
│       │   ├── __init__.py
│       │   └── detector.py
│       ├── llm/
│       │   ├── __init__.py
│       │   └── therapeutic_responder.py
│       ├── safety/
│       │   ├── __init__.py
│       │   └── advisor.py
│       ├── vector_stub/
│       │   ├── __init__.py
│       │   └── stub.py
│       └── voice/
│           ├── __init__.py
│           └── processor.py
└── .zencoder/
    └── rules/
        └── repo.md
```

## Quick Start
1. **Create virtual environment** (already provisioned as `venv/` if desired):
   ```powershell
   python -m venv "c:\Users\KRISHNSA JHA\OneDrive - vitap.ac.in\Desktop\github project\PERSONALIZED_CHATBOT\venv"
   "c:\Users\KRISHNSA JHA\OneDrive - vitap.ac.in\Desktop\github project\PERSONALIZED_CHATBOT\venv\Scripts\activate"
   ```
2. **Install dependencies**:
   ```powershell
   pip install -e .
   ```
3. **Run development server**:
   ```powershell
   flask --app chatbot.wsgi run --debug
   ```
4. **Test endpoints** (PowerShell example):
   ```powershell
   Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:5000/api/v1/chat" -Body (@{user_id="demo"; message="I'm feeling sad today"} | ConvertTo-Json) -ContentType "application/json"
   ```

## Chat Response Payload
The `/api/v1/chat` endpoint returns a JSON payload with the following structure:

- **reply**: String containing the final LLM response sent to the user.
- **emotions**: List of detected emotion labels (e.g., `sadness`, `anger`, `neutral`).
- **risk_level**: One of `low`, `moderate`, or `high`, derived from the detector output.
- **safety_actions**: List of internal safety actions applied (e.g., `append_disclaimer`).
- **retrieved_context**: List of context snippets returned from personalized retrieval.
- **metadata**: Diagnostic block containing confidence signals.
  - **emotion_confidence**: Mapping of emotion labels to per-emotion confidence scores.
  - **risk_confidence**: Aggregated risk score in the range `[0.0, 1.0]`.
  - **detector_trace**: Optional trace data describing detector heuristics. This field is only present when the `expose_detector_trace` feature flag is enabled in the safety advisor configuration.
- **safety**: Always-present block with three keys.
  - **disclaimer**: Safety disclaimer string (empty string when no disclaimer applies).
  - **guidance**: Array of guidance or grounding suggestions surfaced by the safety advisor.
  - **escalation_contacts**: Array of crisis contact links or phone numbers (empty list when not applicable).
- **coaching**: Copy of the guidance messages intended for client UIs that surface coaching tips separately.

## Vector Database Placeholder
- `src/chatbot/vector_stub/stub.py` contains `VectorRetrievalStub`, which returns canned context snippets. Your collaborator can replace this with real vector database logic without changing the pipeline API.

## Next Steps
- Introduce automated schema coverage (unit or contract tests) to validate the enriched `/api/v1/chat` payload.
- Continue refining confidence-driven escalation heuristics, including emotion-specific guidance playbooks.
- Integrate production-grade emotion detection and speech pipelines to replace current stubs.
- Replace the response stub with calls to a therapeutic-tuned LLM.
- Expand the safety advisor with concrete escalation workflows, hotline data, and localization support.
- Add tests under `tests/` once implementation details solidify.