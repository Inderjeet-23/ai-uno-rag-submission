from pathlib import Path

from app.chunker import Chunk
from app.vector_store import FaissVectorStore


def test_vector_store_build_search_save_load(tmp_path: Path) -> None:
    chunks = [
        Chunk("c1", "email_001", "A", "hello world", 2, 0),
        Chunk("c2", "email_002", "B", "budget planning", 2, 0),
    ]
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    store = FaissVectorStore()
    store.build(chunks, embeddings)

    result = store.search([0.9, 0.1, 0.0], top_k=1)
    assert len(result) == 1
    assert result[0]["chunk_id"] == "c1"

    index_path = tmp_path / "faiss.index"
    metadata_path = tmp_path / "chunks.jsonl"
    store.save(index_path, metadata_path)

    loaded = FaissVectorStore()
    loaded.load(index_path, metadata_path)
    result2 = loaded.search([0.0, 1.0, 0.0], top_k=1)
    assert result2[0]["chunk_id"] == "c2"
