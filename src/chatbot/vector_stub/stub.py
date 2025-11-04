"""Vector retrieval utilities with optional ChromaDB integration."""
from __future__ import annotations

import logging
import os
from typing import List, Sequence

from ..config import get_bool_env

VectorDoc = str
VectorResult = List[VectorDoc]
logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency guard
    from ..vector_store.chroma_store import ChromaVectorStore  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
    ChromaVectorStore = None  # type: ignore[assignment]


_DEFAULT_STRATEGIES: Sequence[str] = (
    "Practice deep breathing: Inhale for 4 counts, hold for 4, exhale for 4.",
    "Journaling can help process emotions and gain clarity.",
    "Listening to calming music can help you relax and reset.",
    "Try a five-minute grounding exercise: name five things you can see, four you can touch, three you can hear, two you can smell, one you can taste.",
    "Reach out to someone you trust and share how you're feeling—connection matters.",
)


class _InMemoryVectorStore:
    """Simple in-memory fallback when ChromaDB is unavailable."""

    def __init__(self) -> None:
        self._history: list[str] = list(_DEFAULT_STRATEGIES)

    def add(self, text: str) -> None:
        self._history.append(text)

    def query(self, query: str, n_results: int = 3) -> VectorResult:
        return self._history[-n_results:]

    def count(self) -> int:
        return len(self._history)


class VectorRetrievalStub:
    """Vector retrieval class using ChromaDB when available, in-memory fallback otherwise."""

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or os.getenv("CHROMA_DB_PATH", "./vector_store")
        self._fallback_store = _InMemoryVectorStore()
        self._chroma_store = None

        if not get_bool_env("ENABLE_CHROMADB", True):
            logger.info("ChromaDB integration disabled via environment; using in-memory fallback.")
            return

        if ChromaVectorStore is None:
            logger.warning(
                "Chroma vector store dependencies missing; using in-memory vector store fallback."
            )
            return

        try:
            self._chroma_store = ChromaVectorStore(path=self.db_path)
            self._chroma_store.ensure_seed(_DEFAULT_STRATEGIES)
            logger.info("Chroma vector store initialized at path %s", self.db_path)
        except Exception as exc:  # pragma: no cover - initialization guard
            logger.warning(
                "Failed to initialize Chroma vector store (%s); falling back to in-memory store.",
                exc,
            )
            self._chroma_store = None

    def _active_store(self) -> _InMemoryVectorStore | ChromaVectorStore:
        return self._chroma_store or self._fallback_store

    def add_user_context(self, user_id: str, text: str) -> None:
        """Add personalized context for a user."""

        store = self._active_store()
        if isinstance(store, _InMemoryVectorStore):
            store.add(text)
            return

        doc_id = f"{user_id}_{store.count()}"
        store.add(doc_id=doc_id, text=text)

    def fetch_personalized_context(self, user_id: str, query: str, n_results: int = 3) -> VectorResult:
        """Fetch relevant context snippets for the user and query."""

        store = self._active_store()
        if isinstance(store, _InMemoryVectorStore):
            return store.query(query, n_results=n_results)

        results = store.query(text=query, n_results=n_results)
        return results.documents if results.documents else list(_DEFAULT_STRATEGIES[:n_results])