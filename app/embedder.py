from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib

import numpy as np

from openai import OpenAI


class BaseEmbedder(ABC):
    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError


class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, model: str, api_key: str | None = None) -> None:
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model, input=[text])
        return response.data[0].embedding


class DeterministicEmbedder(BaseEmbedder):
    """Offline fallback embedder based on deterministic text hashing."""

    def __init__(self, dimension: int = 256) -> None:
        self.dimension = dimension

    def _vectorize(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big", signed=False)
        rng = np.random.default_rng(seed)
        vec = rng.normal(0, 1, self.dimension).astype("float32")
        return vec.tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._vectorize(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vectorize(text)
