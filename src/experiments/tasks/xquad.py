"""XQuAD: extractive QA. Metrics: F1 + EM (token-overlap based)."""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any


def _normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _f1(pred: str, gold: str) -> float:
    p_tokens = _normalize_answer(pred).split()
    g_tokens = _normalize_answer(gold).split()
    if not p_tokens or not g_tokens:
        return float(p_tokens == g_tokens)
    common = Counter(p_tokens) & Counter(g_tokens)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(p_tokens)
    recall = n_common / len(g_tokens)
    return 2 * precision * recall / (precision + recall)


def _em(pred: str, gold: str) -> float:
    return float(_normalize_answer(pred) == _normalize_answer(gold))


class XQuADTask:
    name = "xquad"
    metric = "f1_em"

    def load(self, language: str, *, split: str = "validation",
             max_examples: int | None = None) -> list[dict[str, Any]]:
        from datasets import load_dataset

        ds = load_dataset("xquad", f"xquad.{language}", split=split)
        if max_examples:
            ds = ds.select(range(min(len(ds), max_examples)))
        out = []
        for row in ds:
            answers = row["answers"]["text"]
            ans = answers[0] if answers else ""
            out.append({
                "context": row["context"],
                "question": row["question"],
                "answer": ans,
                "all_answers": answers,
            })
        return out

    def render_demo(self, ex: dict[str, Any], language: str) -> str:
        if language == "zh":
            return f"段落：{ex['context']}\n问题：{ex['question']}\n答案：{ex['answer']}"
        return f"Passage: {ex['context']}\nQuestion: {ex['question']}\nAnswer: {ex['answer']}"

    def render_query(self, ex: dict[str, Any], language: str) -> str:
        if language == "zh":
            return f"段落：{ex['context']}\n问题：{ex['question']}\n答案："
        return f"Passage: {ex['context']}\nQuestion: {ex['question']}\nAnswer:"

    def parse_answer(self, completion: str, language: str) -> str:
        first_line = completion.strip().splitlines()[0] if completion.strip() else ""
        return first_line.strip().rstrip(".。")

    def score(self, predicted: Any, gold: Any, language: str) -> dict[str, float]:
        if isinstance(gold, list):
            f1s = [_f1(predicted, g) for g in gold] or [0.0]
            ems = [_em(predicted, g) for g in gold] or [0.0]
            return {"f1": max(f1s), "em": max(ems), "accuracy": max(ems)}
        return {"f1": _f1(predicted, gold or ""), "em": _em(predicted, gold or ""),
                "accuracy": _em(predicted, gold or "")}
