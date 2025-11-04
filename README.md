# Personalized, Emotion-Aware Conversational Assistant

Prototype implementation of a mental-health-oriented chatbot that blends emotion detection, retrieval-augmented responses, safety guardrails, and optional voice processing.

## Features at a Glance
- **Flask API** exposing `/api/v1/health`, `/api/v1/chat`, and `/api/v1/history/<user_id>`.
- **Config helpers** (`get_env`, `get_bool_env`, `get_float_env`) that surface missing configuration early.
- **Emotion detection** driven by Transformers with a keyword-based fallback when ML models are unavailable.
- **Therapeutic LLM responder** using Google Gemini (with graceful fallback messaging and dependency guards).
- **Safety advisor** that appends disclaimers, surfaces crisis guidance, and exposes detector traces when enabled.
- **Voice processing** with optional Whisper transcription and energy-heuristic emotion cues.
- **Vector store layer** that prefers ChromaDB + SentenceTransformers, but falls back to an in-memory store when dependencies are missing.
- **SQLite persistence** tracking conversation turns for lightweight history retrieval.

## Project Layout
```text
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
│       ├── config.py
│       ├── emotion/
│       │   ├── __init__.py
│       │   └── detector.py
│       ├── llm/
│       │   ├── __init__.py
│       │   ├── client.py
│       │   └── therapeutic_responder.py
│       ├── routes/
│       │   └── v1/
│       │       ├── __init__.py
│       │       └── endpoints.py
│       ├── safety/
│       │   ├── __init__.py
│       │   └── advisor.py
│       ├── services/
│       │   ├── chat_pipeline.py
│       │   └── db.py
│       ├── vector_stub/
│       │   ├── __init__.py
│       │   └── stub.py
│       └── voice/
│           ├── __init__.py
│           └── processor.py
├── tests/
│   └── test_chat_payload.py
└── .zencoder/
    └── rules/
        └── repo.md
```

## Prerequisites
- Python **3.10+**
- Optional heavy dependencies (installed automatically when available):
  - `transformers`, `torch`, and `sentence-transformers` for emotion detection and vector encoding.
  - `google-generativeai` to access Gemini.
  - `openai-whisper`, `soundfile`, and `librosa` for voice transcription.
  - `chromadb` for persistent vector retrieval.

If any dependency is missing, the application falls back to lightweight keyword or in-memory implementations with helpful logging.

## Quick Start
1. **Create and activate a virtual environment** (or reuse the existing `venv/`):
   ```powershell
   python -m venv "c:\Users\KRISHNSA JHA\OneDrive - vitap.ac.in\Desktop\github project\PERSONALIZED_CHATBOT\venv"
   "c:\Users\KRISHNSA JHA\OneDrive - vitap.ac.in\Desktop\github project\PERSONALIZED_CHATBOT\venv\Scripts\Activate.ps1"
   ```
2. **Install dependencies** in editable mode:
   ```powershell
   pip install -e .
   ```
3. **Provision environment variables** by creating a `.env` file:
   ```ini
   GEMINI_API_KEY=replace-with-your-key
   GEMINI_MODEL_NAME=models/gemini-1.5-pro
   CHROMA_DB_PATH=c:/path/to/vector_store
   ```
   > Do **not** commit the `.env` file. The repository now includes rules to ignore it.
4. **Run the development server**:
   ```powershell
   flask --app chatbot.wsgi run --debug
   ```
5. **Hit the chat endpoint**:
   ```powershell
   $payload = @{ user_id = "demo"; message = "I'm feeling sad today" } | ConvertTo-Json
   Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:5000/api/v1/chat" -Body $payload -ContentType "application/json"
   ```

## API Overview
### `GET /api/v1/health`
Returns a simple `{ "status": "ok" }` payload.

### `POST /api/v1/chat`
**Body schema**
```json
{
  "user_id": "string (default: anonymous)",
  "message": "required user message",
  "modalities": ["text", "audio"],
  "audio_base64": "optional base64-encoded audio",
  "face_image_b64": "optional base64-encoded image"
}
```
**Response structure**
- **reply**: LLM-generated therapeutic reply (or supportive fallback text).
- **emotions**: List of detected emotion labels.
- **risk_level**: `low`, `moderate`, or `high`.
- **safety_actions**: Internal guardrail actions applied.
- **retrieved_context**: Personalized snippets injected into the prompt.
- **metadata**: Includes emotion/risk confidences and optional detector trace.
- **safety**: Contains disclaimers, guidance, and escalation contacts.
- **coaching**: Guidance tips suitable for separate display.

### `GET /api/v1/history/<user_id>`
Returns the last 20 conversation turns stored for the given user.

## Testing
Run the contract tests to verify payload shape and safety hooks:
```powershell
pytest
```

## Deployment Considerations
- **Secrets**: Store `GEMINI_API_KEY` and any future credentials in a secure secret manager.
- **Vector store**: For production, point `CHROMA_DB_PATH` to a managed or persistent storage location.
- **Heavy models**: Ensure hosts have sufficient GPU/CPU resources when enabling Whisper and Transformer pipelines.
- **Logging**: Integrate structured logging to capture safety events, LLM errors, and fallback usage.

## Roadmap Ideas
- Integrate real vector databases (Pinecone, Weaviate) and a feature store for personalization.
- Replace keyword/energy heuristics with dedicated speech emotion recognition models.
- Layer in escalation workflows tailored to different regions or languages.
- Expand test coverage with integration tests and synthetic conversation scripts.
- Add frontend clients (web/mobile) for end-to-end user testing.

## Contributing
1. Fork the repository and create a feature branch.
2. Keep `.env` and other secrets out of version control.
3. Add unit/integration tests where feasible.
4. Submit a pull request describing the change and safety considerations.

Stay kind and build safely. 😊