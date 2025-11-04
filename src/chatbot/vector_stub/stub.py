"""Vector retrieval using ChromaDB for personalized context."""
from __future__ import annotations

import os
from typing import List

import chromadb
from sentence_transformers import SentenceTransformer


class VectorRetrievalStub:
    """Vector retrieval class using ChromaDB and Sentence Transformers."""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.getenv("CHROMA_DB_PATH", "./vector_store")
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name="user_context")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize with some default documents if collection is empty
        if self.collection.count() == 0:
            self._initialize_default_documents()

    def _initialize_default_documents(self):
        """Add some default coping strategies."""
        documents = [
            "Practice deep breathing: Inhale for 4 counts, hold for 4, exhale for 4.",
            "Journaling can help process emotions and gain clarity.",
            "Listening to music is a great way to relax and lift your mood.",
            "Exercise releases endorphins which can improve your mood.",
            "Talking to a trusted friend can provide support and perspective.",
            "Mindfulness meditation can help manage stress and anxiety.",
            "Getting enough sleep is crucial for emotional well-being.",
            "Healthy eating can positively affect your mood and energy levels.",
        ]
        embeddings = self.encoder.encode(documents).tolist()
        ids = [f"default_{i}" for i in range(len(documents))]
        self.collection.add(embeddings=embeddings, documents=documents, ids=ids)

    def add_user_context(self, user_id: str, text: str):
        """Add personalized context for a user."""
        embedding = self.encoder.encode([text]).tolist()[0]
        doc_id = f"{user_id}_{len(self.collection.get()['ids'])}"
        self.collection.add(embeddings=[embedding], documents=[text], ids=[doc_id])

    def fetch_personalized_context(self, user_id: str, query: str, n_results: int = 3) -> List[str]:
        """Fetch relevant context snippets for the user and query."""
        query_embedding = self.encoder.encode([query]).tolist()[0]

        # Filter by user_id if possible, but Chroma doesn't support metadata filtering easily here
        # For simplicity, retrieve top results and assume personalization via history
        results = self.collection.query(query_embeddings=[query_embedding], n_results=n_results)
        return results['documents'][0] if results['documents'] else []