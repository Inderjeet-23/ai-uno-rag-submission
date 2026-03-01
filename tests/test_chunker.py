from app.chunker import TextChunker


def test_chunker_creates_multiple_chunks_with_overlap() -> None:
    text = " ".join([f"token{i}" for i in range(120)])
    chunker = TextChunker(chunk_size_tokens=50, overlap_tokens=10)

    chunks = chunker.chunk_document(email_id="email_001", subject="Subject", text=text)

    assert len(chunks) >= 3
    assert chunks[0].chunk_id == "email_001_chunk_000"
    assert chunks[1].chunk_id == "email_001_chunk_001"
    assert chunks[0].token_count <= 50
    assert chunks[-1].token_count > 0
