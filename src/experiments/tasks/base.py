"""Task registry for fixed-context experiments."""

from __future__ import annotations


def build_task(name: str):
    name = name.lower()
    if name == "xnli":
        from .xnli import XNLITask

        return XNLITask()
    if name == "xquad":
        from .xquad import XQuADTask

        return XQuADTask()
    if name in ("xlsum", "summarization"):
        from .summarization import SummarizationTask

        return SummarizationTask()
    raise ValueError(f"Unknown task: {name!r}")
