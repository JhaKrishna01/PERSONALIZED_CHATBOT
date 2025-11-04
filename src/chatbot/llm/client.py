"""Google Gemini client utilities."""
from __future__ import annotations

from typing import Any

import google.generativeai as genai

from ..config import get_env


def configure_gemini_client() -> None:
    """Configure the global Gemini client using environment variables."""

    api_key = get_env("GEMINI_API_KEY", required=True)
    # ``configure`` is idempotent; calling it repeatedly updates credentials.
    genai.configure(api_key=api_key)


def get_generative_model(model_name: str) -> genai.GenerativeModel:
    """Instantiate a Gemini generative model after ensuring client configuration."""

    configure_gemini_client()
    return genai.GenerativeModel(model_name=model_name)