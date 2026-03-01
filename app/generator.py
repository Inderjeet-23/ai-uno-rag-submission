from __future__ import annotations

from abc import ABC, abstractmethod

from openai import OpenAI


def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    context_lines = []
    for item in retrieved_chunks:
        context_lines.append(
            f"[chunk_id={item['chunk_id']}, email_id={item['email_id']}, score={item['score']:.4f}]\n{item['text']}"
        )
    context = "\n\n".join(context_lines)
    return (
        "You are a helpful assistant answering questions over internal emails. "
        "Use only the provided context. "
        "If context is insufficient, explicitly say: 'I don't have enough context to answer confidently.' "
        "Cite chunk IDs you used.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{context}\n\n"
        "Return:\n"
        "1) Answer\n"
        "2) Citations: comma-separated chunk IDs"
    )


class BaseAnswerGenerator(ABC):
    @abstractmethod
    def generate(self, question: str, retrieved_chunks: list[dict]) -> str:
        raise NotImplementedError


class OpenAIAnswerGenerator(BaseAnswerGenerator):
    def __init__(self, model: str, api_key: str | None = None) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, question: str, retrieved_chunks: list[dict]) -> str:
        prompt = build_prompt(question=question, retrieved_chunks=retrieved_chunks)
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            temperature=0.2,
        )
        return response.output_text.strip()


class HeuristicAnswerGenerator(BaseAnswerGenerator):
    """Offline fallback answer generator using retrieved chunks."""

    def generate(self, question: str, retrieved_chunks: list[dict]) -> str:
        if not retrieved_chunks:
            return "I don't have enough context to answer confidently."

        top = retrieved_chunks[0]
        snippet = top["text"].strip().replace("\n", " ")
        if len(snippet) > 300:
            snippet = snippet[:297] + "..."

        citations = ", ".join(item["chunk_id"] for item in retrieved_chunks[:3])
        return (
            f"Based on retrieved email context, the most relevant information is: {snippet}\n"
            f"Citations: {citations}"
        )
