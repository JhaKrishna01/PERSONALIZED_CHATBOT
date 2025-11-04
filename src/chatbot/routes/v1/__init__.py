"""Version 1 API routes for the chatbot."""
from __future__ import annotations

from flask import Blueprint, Flask

from .endpoints import register_endpoints


def register_v1_blueprint(app: Flask) -> None:
    """Attach version 1 routes to the provided Flask app."""
    api_v1 = Blueprint("api_v1", __name__, url_prefix="/api/v1")

    register_endpoints(api_v1)
    app.register_blueprint(api_v1)