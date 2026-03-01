from fastapi.testclient import TestClient

from app.main import app


class FakeRAG:
    def __init__(self) -> None:
        self.loaded = True

    def index_loaded(self) -> bool:
        return self.loaded

    def config_snapshot(self) -> dict:
        return {
            "embedding_model": "fake-embed",
            "generation_model": "fake-gen",
            "chunk_size_tokens": 100,
            "chunk_overlap_tokens": 20,
            "top_k_default": 5,
        }

    def build_index(self) -> dict:
        return {
            "indexed_emails": 100,
            "indexed_chunks": 150,
            "duration_ms": 100,
            "faiss_index_path": "artifacts/faiss.index",
            "metadata_path": "artifacts/chunks.jsonl",
        }

    def ask(self, question: str, top_k: int | None = None, debug: bool = False) -> dict:
        payload = {
            "answer": f"Answer for: {question}",
            "citations": [
                {
                    "email_id": "email_001",
                    "email_number": "001",
                    "chunk_id": "email_001_chunk_000",
                    "score": 0.99,
                    "subject": "Project Update",
                }
            ],
            "latency_ms": 12,
            "cited_emails": [
                {
                    "email_id": "email_001",
                    "email_number": "001",
                    "subject": "Project Update",
                    "full_text": "Subject: Project Update\\n\\nFrom: A <a@x.com>\\nTo: B <b@x.com>\\n\\nFull email body.",
                }
            ],
        }
        if debug:
            payload["retrieved_context"] = [{"text": "sample"}]
        return payload


def test_health_endpoint() -> None:
    app.state.rag = FakeRAG()
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ui_endpoint() -> None:
    app.state.rag = FakeRAG()
    client = TestClient(app)

    response = client.get("/")
    assert response.status_code == 200
    assert "Mini RAG Email QA" in response.text


def test_config_endpoint() -> None:
    app.state.rag = FakeRAG()
    client = TestClient(app)

    response = client.get("/config")
    assert response.status_code == 200
    assert response.json()["embedding_model"] == "fake-embed"


def test_index_endpoint() -> None:
    app.state.rag = FakeRAG()
    client = TestClient(app)

    response = client.post("/index")
    assert response.status_code == 200
    assert response.json()["indexed_emails"] == 100


def test_ask_endpoint() -> None:
    app.state.rag = FakeRAG()
    client = TestClient(app)

    response = client.post("/ask", json={"question": "What is the project status?", "debug": True})
    assert response.status_code == 200
    data = response.json()
    assert "Answer for" in data["answer"]
    assert data["citations"][0]["chunk_id"] == "email_001_chunk_000"
    assert data["retrieved_context"][0]["text"] == "sample"
