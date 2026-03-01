from __future__ import annotations

from .embedder import BaseEmbedder
from .vector_store import FaissVectorStore


class Retriever:
    def __init__(self, embedder: BaseEmbedder, vector_store: FaissVectorStore) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, question: str, top_k: int = 5) -> list[dict]:
        query_embedding = self.embedder.embed_query(question)
        return self.vector_store.search(query_embedding=query_embedding, top_k=top_k)
