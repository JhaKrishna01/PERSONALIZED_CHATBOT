"""Vector retrieval utilities with optional ChromaDB integration."""
from __future__ import annotations

import logging
import os
from typing import List, Sequence

VectorDoc = str
VectorResult = List[VectorDoc]
logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency guard
    import chromadb  # type: ignore
    from sentence_transformers import SentenceTransformer  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
    chromadb = None  # type: ignore[assignment]
    SentenceTransformer = None  # type: ignore[assignment]


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

        if chromadb is None or SentenceTransformer is None:
            logger.warning(
                "ChromaDB or SentenceTransformers not available; using in-memory vector store fallback."
            )
            self._store: _InMemoryVectorStore | None = _InMemoryVectorStore()
            self._client = None
            self._collection = None
            self._encoder = None
        else:
            self._store = None
            self._client = chromadb.PersistentClient(path=self.db_path)
            self._collection = self._client.get_or_create_collection(name="user_context")
            self._encoder = SentenceTransformer("all-MiniLM-L6-v2")

            if self._collection.count() == 0:
                self._initialize_default_documents()

    def _initialize_default_documents(self) -> None:
        """Seed the vector store with coping strategies."""

        if self._store is not None:
            for doc in _DEFAULT_STRATEGIES:
                self._store.add(doc)
            return

        assert self._collection is not None and self._encoder is not None

        embeddings = self._encoder.encode(list(_DEFAULT_STRATEGIES)).tolist()
        ids = [f"default_{i}" for i in range(len(_DEFAULT_STRATEGIES))]
        self._collection.add(embeddings=embeddings, documents=list(_DEFAULT_STRATEGIES), ids=ids)

    def add_user_context(self, user_id: str, text: str) -> None:
        """Add personalized context for a user."""

        if self._store is not None:
            self._store.add(text)
            return

        if self._collection is None or self._encoder is None:
            logger.warning("Vector store is not initialized; skipping context storage.")
            return

        embedding = self._encoder.encode([text]).tolist()[0]
        doc_id = f"{user_id}_{self._collection.count()}"
        self._collection.add(embeddings=[embedding], documents=[text], ids=[doc_id])

    def fetch_personalized_context(self, user_id: str, query: str, n_results: int = 3) -> VectorResult:
        """Fetch relevant context snippets for the user and query."""

        if self._store is not None:
            return self._store.query(query, n_results=n_results)

        if self._collection is None or self._encoder is None:
            logger.warning("Vector store is not initialized; returning default strategies.")
            return list(_DEFAULT_STRATEGIES[:n_results])

        query_embedding = self._encoder.encode([query]).tolist()[0]
        results = self._collection.query(query_embeddings=[query_embedding], n_results=n_results)
        return results["documents"][0] if results.get("documents") else []