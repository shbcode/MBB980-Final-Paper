"""Optional XL-Sum-style summarization. Uses ROUGE-L."""

from __future__ import annotations

from typing import Any


class SummarizationTask:
    name = "xlsum"
    metric = "rougeL"

    def load(self, language: str, *, split: str = "validation",
             max_examples: int | None = None) -> list[dict[str, Any]]:
        from datasets import load_dataset

        cfg = "english" if language == "en" else "chinese_simplified"
        ds = load_dataset("csebuetnlp/xlsum", cfg, split=split)
        if max_examples:
            ds = ds.select(range(min(len(ds), max_examples)))
        out = []
        for row in ds:
            out.append({
                "text": row["text"],
                "summary": row["summary"],
                "answer": row["summary"],
            })
        return out

    def render_demo(self, ex: dict[str, Any], language: str) -> str:
        if language == "zh":
            return f"文章：{ex['text']}\n摘要：{ex['summary']}"
        return f"Article: {ex['text']}\nSummary: {ex['summary']}"

    def render_query(self, ex: dict[str, Any], language: str) -> str:
        if language == "zh":
            return f"文章：{ex['text']}\n摘要："
        return f"Article: {ex['text']}\nSummary:"

    def parse_answer(self, completion: str, language: str) -> str:
        return completion.strip()

    def score(self, predicted: Any, gold: Any, language: str) -> dict[str, float]:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        s = scorer.score(target=gold or "", prediction=predicted or "")
        return {"rougeL": float(s["rougeL"].fmeasure)}
