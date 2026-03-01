from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from .chunker import Chunk


class FaissVectorStore:
    def __init__(self) -> None:
        self.index: faiss.IndexFlatIP | None = None
        self.metadata: list[dict] = []

    @staticmethod
    def _normalize(embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms

    def build(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks or not embeddings:
            raise ValueError("Cannot build index without chunks and embeddings")

        vectors = np.array(embeddings, dtype="float32")
        vectors = self._normalize(vectors)
        dim = vectors.shape[1]

        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)
        self.metadata = [chunk.to_dict() for chunk in chunks]

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        if self.index is None:
            raise RuntimeError("Index not loaded")

        query = np.array([query_embedding], dtype="float32")
        query = self._normalize(query)
        scores, indices = self.index.search(query, top_k)

        results: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            item = dict(self.metadata[idx])
            item["score"] = float(score)
            results.append(item)
        return results

    def save(self, index_path: Path, metadata_path: Path) -> None:
        if self.index is None:
            raise RuntimeError("Index not loaded")
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        with metadata_path.open("w", encoding="utf-8") as f:
            for row in self.metadata:
                f.write(json.dumps(row) + "\n")

    def load(self, index_path: Path, metadata_path: Path) -> None:
        self.index = faiss.read_index(str(index_path))
        rows: list[dict] = []
        with metadata_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        self.metadata = rows

    def size(self) -> int:
        return len(self.metadata)
