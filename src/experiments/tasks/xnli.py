"""XNLI 3-way classification: entailment / neutral / contradiction.

Examples are loaded via `datasets`. Each example dict carries
{premise, hypothesis, label, answer} so the same shape works for both
demos and queries.
"""

from __future__ import annotations

from typing import Any

LABEL_NAMES_EN = ["entailment", "neutral", "contradiction"]
LABEL_NAMES_ZH = ["蕴含", "中立", "矛盾"]


class XNLITask:
    name = "xnli"
    metric = "accuracy"

    def load(self, language: str, *, split: str = "validation",
             max_examples: int | None = None) -> list[dict[str, Any]]:
        from datasets import load_dataset

        ds = load_dataset("xnli", language, split=split)
        if max_examples:
            ds = ds.select(range(min(len(ds), max_examples)))
        out = []
        for row in ds:
            ans = LABEL_NAMES_EN[row["label"]]  # canonical EN label
            out.append({
                "premise": row["premise"],
                "hypothesis": row["hypothesis"],
                "label": row["label"],
                "answer": ans,
            })
        return out

    def render_demo(self, ex: dict[str, Any], language: str) -> str:
        if language == "zh":
            return (
                f"前提：{ex['premise']}\n"
                f"假设：{ex['hypothesis']}\n"
                f"关系：{LABEL_NAMES_ZH[ex['label']]}"
            )
        return (
            f"Premise: {ex['premise']}\n"
            f"Hypothesis: {ex['hypothesis']}\n"
            f"Relation: {LABEL_NAMES_EN[ex['label']]}"
        )

    def render_query(self, ex: dict[str, Any], language: str) -> str:
        if language == "zh":
            return (
                f"前提：{ex['premise']}\n"
                f"假设：{ex['hypothesis']}\n"
                f"关系："
            )
        return (
            f"Premise: {ex['premise']}\n"
            f"Hypothesis: {ex['hypothesis']}\n"
            f"Relation:"
        )

    def parse_answer(self, completion: str, language: str) -> str:
        text = completion.strip().lower()
        # First-token-of-line heuristic.
        first_line = text.splitlines()[0] if text else ""
        if language == "zh":
            for zh, en in zip(LABEL_NAMES_ZH, LABEL_NAMES_EN, strict=True):
                if zh in first_line or en.lower() in first_line:
                    return en
        else:
            for en in LABEL_NAMES_EN:
                if en in first_line:
                    return en
        return first_line.split()[0] if first_line else ""

    def score(self, predicted: Any, gold: Any, language: str) -> dict[str, float]:
        return {"accuracy": 1.0 if predicted == gold else 0.0}
