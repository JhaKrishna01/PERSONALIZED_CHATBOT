"""ChromaDB-backed vector store service."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import chromadb
from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class QueryResult:
    """Typed structure for Chroma query results."""

    documents: list[str]


class ChromaVectorStore:
    """Persistent vector store for personalized context."""

    def __init__(self, path: str, collection_name: str = "user_context") -> None:
        self._client = chromadb.PersistentClient(path=path)
        self._collection = self._client.get_or_create_collection(name=collection_name)
        self._encoder = SentenceTransformer("all-MiniLM-L6-v2")

    def ensure_seed(self, defaults: Sequence[str]) -> None:
        if self._collection.count() > 0:
            return

        embeddings = self._encoder.encode(list(defaults)).tolist()
        ids = [f"default_{idx}" for idx, _ in enumerate(defaults)]
        self._collection.add(embeddings=embeddings, documents=list(defaults), ids=ids)

    def count(self) -> int:
        """Return the total number of stored documents."""

        return self._collection.count()

    def add(self, doc_id: str, text: str) -> None:
        embedding = self._encoder.encode([text]).tolist()[0]
        self._collection.add(embeddings=[embedding], documents=[text], ids=[doc_id])

    def query(self, text: str, n_results: int) -> QueryResult:
        embedding = self._encoder.encode([text]).tolist()[0]
        results = self._collection.query(query_embeddings=[embedding], n_results=n_results)
        documents = results.get("documents") or []
        first_page = documents[0] if documents else []
        return QueryResult(documents=list(first_page))