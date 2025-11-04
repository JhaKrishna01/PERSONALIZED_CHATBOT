"""Flask application factory for the personalized mental health chatbot."""
from __future__ import annotations

from flask import Flask

from .routes.v1 import register_v1_blueprint


def create_app() -> Flask:
    """Initialize the Flask application with registered blueprints."""
    app = Flask(__name__)

    # Register API blueprints
    register_v1_blueprint(app)

    @app.get("/")
    def root() -> tuple[dict[str, str], int]:
        """Provide a minimal landing response for the API root."""
        return {"message": "Chatbot API is online", "docs": "/api/v1/health"}, 200

    return app