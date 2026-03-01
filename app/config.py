from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("emails")
    artifact_dir: Path = Path("artifacts")
    faiss_index_path: Path = Path("artifacts/faiss.index")
    metadata_path: Path = Path("artifacts/chunks.jsonl")
    manifest_path: Path = Path("artifacts/index_manifest.json")

    embedding_model: str = "text-embedding-3-small"
    generation_model: str = "gpt-4o-mini"

    chunk_size_tokens: int = 220
    chunk_overlap_tokens: int = 40
    top_k_default: int = 5

    openai_api_key: str | None = None

    @staticmethod
    def from_env() -> "Settings":
        api_key = os.getenv("OPENAI_API_KEY")
        return Settings(
            data_dir=Path(os.getenv("DATA_DIR", "emails")),
            artifact_dir=Path(os.getenv("ARTIFACT_DIR", "artifacts")),
            faiss_index_path=Path(os.getenv("FAISS_INDEX_PATH", "artifacts/faiss.index")),
            metadata_path=Path(os.getenv("CHUNKS_METADATA_PATH", "artifacts/chunks.jsonl")),
            manifest_path=Path(os.getenv("INDEX_MANIFEST_PATH", "artifacts/index_manifest.json")),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            generation_model=os.getenv("GENERATION_MODEL", "gpt-4o-mini"),
            chunk_size_tokens=int(os.getenv("CHUNK_SIZE_TOKENS", "220")),
            chunk_overlap_tokens=int(os.getenv("CHUNK_OVERLAP_TOKENS", "40")),
            top_k_default=int(os.getenv("TOP_K_DEFAULT", "5")),
            openai_api_key=api_key,
        )

    def ensure_artifact_dir(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
