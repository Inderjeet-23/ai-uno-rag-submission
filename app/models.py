from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, description="User question")
    top_k: int | None = Field(default=None, ge=1, le=20)
    debug: bool = False


class Citation(BaseModel):
    email_id: str
    email_number: str
    chunk_id: str
    score: float
    subject: str


class CitedEmail(BaseModel):
    email_id: str
    email_number: str
    subject: str
    full_text: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    latency_ms: int
    retrieved_context: list[dict] | None = None
    cited_emails: list[CitedEmail] | None = None


class IndexResponse(BaseModel):
    indexed_emails: int
    indexed_chunks: int
    duration_ms: int
    faiss_index_path: str
    metadata_path: str


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool


class ConfigResponse(BaseModel):
    embedding_model: str
    generation_model: str
    chunk_size_tokens: int
    chunk_overlap_tokens: int
    top_k_default: int
