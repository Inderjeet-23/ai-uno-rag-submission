from __future__ import annotations

from dataclasses import asdict, dataclass

try:
    import tiktoken
except ImportError:  # pragma: no cover
    tiktoken = None


@dataclass
class Chunk:
    chunk_id: str
    email_id: str
    subject: str
    text: str
    token_count: int
    chunk_index: int

    def to_dict(self) -> dict:
        return asdict(self)


class TextChunker:
    def __init__(self, chunk_size_tokens: int = 220, overlap_tokens: int = 40) -> None:
        if overlap_tokens >= chunk_size_tokens:
            raise ValueError("overlap_tokens must be smaller than chunk_size_tokens")
        self.chunk_size_tokens = chunk_size_tokens
        self.overlap_tokens = overlap_tokens
        self._encoding = None
        if tiktoken:
            try:
                self._encoding = tiktoken.get_encoding("cl100k_base")
            except Exception:
                # Offline-safe fallback for environments that cannot fetch tokenizer assets.
                self._encoding = None

    def chunk_document(self, email_id: str, subject: str, text: str) -> list[Chunk]:
        stride = self.chunk_size_tokens - self.overlap_tokens
        chunks: list[Chunk] = []
        chunk_index = 0

        if self._encoding:
            token_ids = self._encoding.encode(text)
            for start in range(0, len(token_ids), stride):
                window = token_ids[start : start + self.chunk_size_tokens]
                if not window:
                    continue

                chunk_text = self._encoding.decode(window).strip()
                if not chunk_text:
                    continue

                chunks.append(
                    Chunk(
                        chunk_id=f"{email_id}_chunk_{chunk_index:03d}",
                        email_id=email_id,
                        subject=subject,
                        text=chunk_text,
                        token_count=len(window),
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1

                if start + self.chunk_size_tokens >= len(token_ids):
                    break
            return chunks

        words = text.split()
        for start in range(0, len(words), stride):
            window_words = words[start : start + self.chunk_size_tokens]
            if not window_words:
                continue
            chunk_text = " ".join(window_words).strip()
            chunks.append(
                Chunk(
                    chunk_id=f"{email_id}_chunk_{chunk_index:03d}",
                    email_id=email_id,
                    subject=subject,
                    text=chunk_text,
                    token_count=len(window_words),
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1

            if start + self.chunk_size_tokens >= len(words):
                break

        return chunks
