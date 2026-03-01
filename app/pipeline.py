from __future__ import annotations

import hashlib
import json
import time

from .chunker import Chunk, TextChunker
from .config import Settings
from .email_parser import load_email_documents
from .embedder import BaseEmbedder, DeterministicEmbedder, OpenAIEmbedder
from .generator import BaseAnswerGenerator, HeuristicAnswerGenerator, OpenAIAnswerGenerator
from .retriever import Retriever
from .vector_store import FaissVectorStore


class RAGPipeline:
    def __init__(
        self,
        settings: Settings,
        embedder: BaseEmbedder | None = None,
        answer_generator: BaseAnswerGenerator | None = None,
        vector_store: FaissVectorStore | None = None,
    ) -> None:
        self.settings = settings
        self.settings.ensure_artifact_dir()
        if embedder is not None:
            self.embedder = embedder
        elif settings.openai_api_key:
            self.embedder = OpenAIEmbedder(
                model=settings.embedding_model,
                api_key=settings.openai_api_key,
            )
        else:
            self.embedder = DeterministicEmbedder()

        if answer_generator is not None:
            self.answer_generator = answer_generator
        elif settings.openai_api_key:
            self.answer_generator = OpenAIAnswerGenerator(
                model=settings.generation_model,
                api_key=settings.openai_api_key,
            )
        else:
            self.answer_generator = HeuristicAnswerGenerator()
        self.chunker = TextChunker(
            chunk_size_tokens=settings.chunk_size_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        )
        self.vector_store = vector_store or FaissVectorStore()
        self.retriever = Retriever(self.embedder, self.vector_store)

    def _manifest_payload(self, email_count: int, chunk_count: int) -> dict:
        payload = {
            "email_count": email_count,
            "chunk_count": chunk_count,
            "embedding_model": self.settings.embedding_model,
            "generation_model": self.settings.generation_model,
            "chunk_size_tokens": self.settings.chunk_size_tokens,
            "chunk_overlap_tokens": self.settings.chunk_overlap_tokens,
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        payload["config_hash"] = hashlib.sha256(encoded).hexdigest()
        return payload

    def build_index(self) -> dict:
        start = time.perf_counter()
        docs = load_email_documents(self.settings.data_dir)

        all_chunks: list[Chunk] = []
        for doc in docs:
            chunks = self.chunker.chunk_document(
                email_id=doc.email_id,
                subject=doc.subject,
                text=doc.canonical_text,
            )
            all_chunks.extend(chunks)

        embeddings = self.embedder.embed_texts([chunk.text for chunk in all_chunks])
        self.vector_store.build(chunks=all_chunks, embeddings=embeddings)
        self.vector_store.save(self.settings.faiss_index_path, self.settings.metadata_path)

        manifest = self._manifest_payload(email_count=len(docs), chunk_count=len(all_chunks))
        with self.settings.manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        duration_ms = int((time.perf_counter() - start) * 1000)
        return {
            "indexed_emails": len(docs),
            "indexed_chunks": len(all_chunks),
            "duration_ms": duration_ms,
            "faiss_index_path": str(self.settings.faiss_index_path),
            "metadata_path": str(self.settings.metadata_path),
        }

    def load_index(self) -> bool:
        if not (self.settings.faiss_index_path.exists() and self.settings.metadata_path.exists()):
            return False
        self.vector_store.load(self.settings.faiss_index_path, self.settings.metadata_path)
        return True

    def ask(self, question: str, top_k: int | None = None, debug: bool = False) -> dict:
        start = time.perf_counter()
        use_top_k = top_k or self.settings.top_k_default
        retrieved = self.retriever.retrieve(question, top_k=use_top_k)
        answer = self.answer_generator.generate(question=question, retrieved_chunks=retrieved)

        citations = [
            {
                "email_id": item["email_id"],
                "email_number": item["email_id"].split("_")[-1],
                "chunk_id": item["chunk_id"],
                "score": item["score"],
                "subject": item["subject"],
            }
            for item in retrieved
        ]

        cited_emails = []
        seen_ids = set()
        for item in retrieved:
            email_id = item["email_id"]
            if email_id in seen_ids:
                continue
            seen_ids.add(email_id)
            email_path = self.settings.data_dir / f"{email_id}.txt"
            full_text = ""
            if email_path.exists():
                full_text = email_path.read_text(encoding="utf-8")
            cited_emails.append(
                {
                    "email_id": email_id,
                    "email_number": email_id.split("_")[-1],
                    "subject": item["subject"],
                    "full_text": full_text,
                }
            )

        duration_ms = int((time.perf_counter() - start) * 1000)
        payload = {
            "answer": answer,
            "citations": citations,
            "latency_ms": duration_ms,
            "cited_emails": cited_emails,
        }
        if debug:
            payload["retrieved_context"] = retrieved
        return payload

    def config_snapshot(self) -> dict:
        return {
            "embedding_model": self.settings.embedding_model,
            "generation_model": self.settings.generation_model,
            "chunk_size_tokens": self.settings.chunk_size_tokens,
            "chunk_overlap_tokens": self.settings.chunk_overlap_tokens,
            "top_k_default": self.settings.top_k_default,
        }

    def index_loaded(self) -> bool:
        return self.vector_store.size() > 0
