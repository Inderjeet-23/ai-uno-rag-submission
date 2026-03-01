#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import Settings
from app.pipeline import RAGPipeline


@dataclass
class EvalCase:
    question: str
    expected_keywords: list[str]


EVAL_SET = [
    EvalCase("Which email talks about budget approval and fiscal year planning?", ["budget", "fiscal", "approval"]),
    EvalCase("Find context related to a technical issue in production systems.", ["technical issue", "production", "incident"]),
    EvalCase("What emails mention a meeting request for next quarter strategy?", ["meeting", "quarter", "strategy"]),
    EvalCase("Where is client feedback discussed, including strengths and improvements?", ["client feedback", "positive", "improve"]),
    EvalCase("Locate emails about a deadline extension request.", ["deadline", "extension", "timeline"]),
    EvalCase("Which content is about training opportunities and professional development?", ["training", "workshop", "development"]),
    EvalCase("Find vendor proposal discussions with pricing and terms.", ["vendor", "proposal", "pricing"]),
    EvalCase("Which emails discuss annual performance review meetings?", ["performance review", "annual", "career"]),
    EvalCase("Identify project update emails mentioning milestones and progress.", ["project update", "milestone", "progress"]),
    EvalCase("Find team announcement messages about structure changes.", ["team announcement", "reorganize", "structure"]),
]


def keyword_relevant(text: str, keywords: list[str]) -> bool:
    lower = text.lower()
    return any(keyword.lower() in lower for keyword in keywords)


def compute_retrieval_metrics(rag: RAGPipeline, top_k: int = 5) -> dict:
    hits = 0
    reciprocal_ranks = []
    per_case = []

    for case in EVAL_SET:
        retrieved = rag.retriever.retrieve(case.question, top_k=top_k)
        found_at = None
        for idx, item in enumerate(retrieved, start=1):
            haystack = f"{item.get('subject', '')}\n{item.get('text', '')}"
            if keyword_relevant(haystack, case.expected_keywords):
                found_at = idx
                break

        if found_at is not None:
            hits += 1
            reciprocal_ranks.append(1.0 / found_at)
        else:
            reciprocal_ranks.append(0.0)

        per_case.append(
            {
                "question": case.question,
                "hit": found_at is not None,
                "first_relevant_rank": found_at,
            }
        )

    count = len(EVAL_SET)
    return {
        "recall_at_k": hits / count if count else 0.0,
        "mrr": sum(reciprocal_ranks) / count if count else 0.0,
        "cases": per_case,
    }


def compute_groundedness_score(rag: RAGPipeline, top_k: int = 5) -> dict:
    # Lightweight proxy: ensure generated answers include citation marker line.
    total = 0
    with_citation_line = 0

    for case in EVAL_SET[:5]:
        response = rag.ask(case.question, top_k=top_k, debug=False)
        total += 1
        if "citation" in response["answer"].lower() or "chunk" in response["answer"].lower():
            with_citation_line += 1

    return {
        "cases_evaluated": total,
        "citation_line_ratio": (with_citation_line / total) if total else 0.0,
    }


def main() -> None:
    settings = Settings.from_env()
    rag = RAGPipeline(settings=settings)

    if not rag.load_index():
        rag.build_index()

    retrieval_metrics = compute_retrieval_metrics(rag)

    generation_metrics = compute_groundedness_score(rag)

    results = {
        "retrieval": retrieval_metrics,
        "generation": generation_metrics,
        "notes": [
            "Retrieval metrics use keyword-based relevance proxy over synthetic topics.",
            "Generation metric is a lightweight groundedness proxy.",
        ],
    }

    settings.artifact_dir.mkdir(parents=True, exist_ok=True)
    results_path = settings.artifact_dir / "eval_results.json"
    report_path = settings.artifact_dir / "evaluation_report.md"

    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    report = [
        "# Evaluation Report",
        "",
        f"- Recall@5: {retrieval_metrics['recall_at_k']:.3f}",
        f"- MRR: {retrieval_metrics['mrr']:.3f}",
    ]

    report.append(
        f"- Generation citation-line ratio: {generation_metrics['citation_line_ratio']:.3f} "
        f"(n={generation_metrics['cases_evaluated']})"
    )

    report.extend(
        [
            "",
            "## Method",
            "- Retrieval evaluated with 10 topic-focused questions.",
            "- Relevance uses keyword match in retrieved chunk text/subject.",
            "- This is a proxy metric suitable for assignment-scale benchmarking.",
        ]
    )

    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(json.dumps(results, indent=2))
    print(f"Wrote {results_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
