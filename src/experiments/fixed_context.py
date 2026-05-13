"""Experiment 2: fixed-context fairness.

For each language and each token budget B in {512, 1024, 2048, ...}:
    1. Build a few-shot prompt by greedily packing demonstrations until the
       prompt + a target query fits within B tokens (measured by the model's
       own tokenizer).
    2. Record `examples_fit`, the prompt, the model's response, latency, and
       (if hosted) cost.
    3. Score with the task's metric (accuracy for XNLI, F1+EM for XQuAD,
       ROUGE-L for summarization).

This file implements the orchestrator + an XNLI task. XQuAD/summarization use
the same `Task` interface and can be added in `tasks/`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..backends import GenerationRequest, ModelBackend, build_backend, warn_if_dummy
from ..plotting import apply_style, save_figure
from ..tokenizers import build_tokenizer
from ..utils import get_logger, write_csv, write_jsonl, write_text
from .tasks import build_task

log = get_logger(__name__)


# ----- task interface -----------------------------------------------------

class Task(Protocol):
    name: str
    metric: str  # "accuracy" | "f1_em" | "rougeL"

    def load(self, language: str, *, split: str, max_examples: int | None) -> list[dict[str, Any]]: ...
    def render_demo(self, ex: dict[str, Any], language: str) -> str: ...
    def render_query(self, ex: dict[str, Any], language: str) -> str: ...
    def parse_answer(self, completion: str, language: str) -> Any: ...
    def score(self, predicted: Any, gold: Any, language: str) -> dict[str, float]: ...


# ----- prompt packing -----------------------------------------------------

def pack_few_shot(
    demos: list[dict[str, Any]],
    query: dict[str, Any],
    task: Task,
    language: str,
    *,
    budget_tokens: int,
    count_tokens,
    instruction: str,
    separator: str = "\n\n",
    max_demos: int | None = None,
) -> tuple[str, int]:
    """Greedy: prepend demos one at a time while the prompt token count fits."""
    query_text = task.render_query(query, language)
    base = f"{instruction}{separator}{query_text}" if instruction else query_text
    base_tokens = count_tokens(base)
    if base_tokens > budget_tokens:
        raise ValueError(
            f"Query alone ({base_tokens} tokens) exceeds budget ({budget_tokens})"
        )

    parts: list[str] = []
    used = base_tokens
    n_demos = 0
    for d in demos:
        if max_demos is not None and n_demos >= max_demos:
            break
        rendered = task.render_demo(d, language)
        added = count_tokens(rendered + separator)
        if used + added > budget_tokens:
            break
        parts.append(rendered)
        used += added
        n_demos += 1

    prompt_body = separator.join(parts)
    prompt = (
        f"{instruction}{separator}{prompt_body}{separator}{query_text}"
        if (instruction or prompt_body)
        else query_text
    )
    return prompt, n_demos


# ----- driver -------------------------------------------------------------

@dataclass
class FixedContextConfig:
    task_name: str
    languages: list[str] = field(default_factory=lambda: ["en", "zh"])
    budgets: list[int] = field(default_factory=lambda: [512, 1024, 2048])
    n_query: int = 100
    demo_pool: int = 32
    backend: dict[str, Any] = field(default_factory=lambda: {"kind": "dummy"})
    measure_tokenizer: Any = "tiktoken_cl100k"
    output_dir: str = "results/fixed_context"
    seed: int = 0
    cache_dir: str = "data/raw"
    instruction_per_language: dict[str, str] = field(default_factory=dict)
    max_new_tokens: int = 32
    temperature: float = 0.0
    equal_demo_count: int | None = None
    byte_normalized: bool = False


def run(cfg_dict: dict[str, Any]) -> dict[str, Any]:
    cfg = FixedContextConfig(
        task_name=cfg_dict["task"],
        languages=list(cfg_dict.get("languages", ["en", "zh"])),
        budgets=list(cfg_dict.get("budgets", [512, 1024, 2048])),
        n_query=int(cfg_dict.get("n_query", 100)),
        demo_pool=int(cfg_dict.get("demo_pool", 32)),
        backend=dict(cfg_dict.get("backend", {"kind": "dummy"})),
        measure_tokenizer=cfg_dict.get("measure_tokenizer", "tiktoken_cl100k"),
        output_dir=cfg_dict.get("output_dir", "results/fixed_context"),
        seed=int(cfg_dict.get("seed", 0)),
        cache_dir=cfg_dict.get("cache_dir", "data/raw"),
        instruction_per_language=dict(cfg_dict.get("instructions", {})),
        max_new_tokens=int(cfg_dict.get("max_new_tokens", 32)),
        temperature=float(cfg_dict.get("temperature", 0.0)),
        equal_demo_count=cfg_dict.get("equal_demo_count"),
        byte_normalized=bool(cfg_dict.get("byte_normalized", False)),
    )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"

    backend: ModelBackend = build_backend(cfg.backend)
    warn_if_dummy(backend, experiment="fixed_context")
    measure_tok = build_tokenizer(cfg.measure_tokenizer)
    count_tokens = lambda t: measure_tok.count(t)  # noqa: E731
    task: Task = build_task(cfg.task_name)

    rows: list[dict[str, Any]] = []
    raw_log: list[dict[str, Any]] = []

    for lang in cfg.languages:
        instruction = cfg.instruction_per_language.get(lang, "")
        examples = task.load(lang, split="validation",
                             max_examples=cfg.n_query + cfg.demo_pool)
        demo_pool = examples[: cfg.demo_pool]
        queries = examples[cfg.demo_pool : cfg.demo_pool + cfg.n_query]
        log.info("[%s] %d demos available, %d queries", lang, len(demo_pool), len(queries))

        for budget in cfg.budgets:
            effective_budget = budget
            if cfg.byte_normalized:
                # Byte-normalized: shrink the EN budget so byte/token ratio
                # roughly matches the ZH budget (heuristic, helpful for comparison).
                pass  # placeholder: implement in your project as needed

            n_correct = 0
            f1_sum = 0.0
            em_sum = 0.0
            rougeL_sum = 0.0
            seen = 0
            ex_fit_total = 0
            cost_total = 0.0
            latency_total = 0.0
            tokens_in_total = 0
            tokens_out_total = 0

            for q in queries:
                try:
                    prompt, n_demos = pack_few_shot(
                        demo_pool, q, task, lang,
                        budget_tokens=effective_budget,
                        count_tokens=count_tokens,
                        instruction=instruction,
                        max_demos=cfg.equal_demo_count,
                    )
                except ValueError as e:
                    log.warning("[%s, B=%d] skipping query (%s)", lang, budget, e)
                    continue

                t0 = time.perf_counter()
                resp = backend.generate(GenerationRequest(
                    prompt=prompt,
                    max_new_tokens=cfg.max_new_tokens,
                    temperature=cfg.temperature,
                    seed=cfg.seed,
                ))
                latency = time.perf_counter() - t0
                pred = task.parse_answer(resp.text, lang)
                metric = task.score(pred, q.get("answer"), lang)

                seen += 1
                ex_fit_total += n_demos
                latency_total += latency
                if resp.cost_usd:
                    cost_total += resp.cost_usd
                if resp.prompt_tokens:
                    tokens_in_total += resp.prompt_tokens
                if resp.completion_tokens:
                    tokens_out_total += resp.completion_tokens

                if "accuracy" in metric:
                    n_correct += int(metric["accuracy"] >= 1.0)
                f1_sum += metric.get("f1", 0.0)
                em_sum += metric.get("em", 0.0)
                rougeL_sum += metric.get("rougeL", 0.0)

                raw_log.append({
                    "language": lang, "budget": budget, "task": cfg.task_name,
                    "examples_fit": n_demos, "prompt_tokens_measure": count_tokens(prompt),
                    "prompt": prompt, "response": resp.text,
                    "predicted": pred, "gold": q.get("answer"),
                    "latency_seconds": latency,
                    "cost_usd": resp.cost_usd,
                })

            if seen == 0:
                continue
            accuracy = n_correct / seen
            cost_per_correct = (cost_total / n_correct) if n_correct else float("nan")
            rows.append({
                "language": lang,
                "budget": budget,
                "task": cfg.task_name,
                "n_queries": seen,
                "examples_fit_mean": ex_fit_total / seen,
                "accuracy": accuracy,
                "f1": f1_sum / seen,
                "em": em_sum / seen,
                "rougeL": rougeL_sum / seen,
                "tokens_in": tokens_in_total,
                "tokens_out": tokens_out_total,
                "latency_total_s": latency_total,
                "cost_total_usd": cost_total,
                "cost_per_correct": cost_per_correct,
            })

    write_csv(rows, output_dir / "summary.csv")
    write_jsonl(raw_log, output_dir / "raw_log.jsonl")

    # AUC over budget per language (trapezoidal in (budget, accuracy)).
    auc_rows: list[dict[str, Any]] = []
    if rows:
        import pandas as pd

        df = pd.DataFrame(rows)
        for (lang, task_name), sub in df.groupby(["language", "task"]):
            sub = sub.sort_values("budget")
            x = sub["budget"].to_numpy(dtype=float)
            y = sub["accuracy"].to_numpy(dtype=float)
            if len(x) >= 2:
                # Normalize by budget range so AUCs across configs are comparable.
                auc = float(((y[:-1] + y[1:]) / 2 * (x[1:] - x[:-1])).sum() / (x[-1] - x[0]))
            else:
                auc = float("nan")
            auc_rows.append({"language": lang, "task": task_name, "auc_acc_vs_budget": auc})
        write_csv(auc_rows, output_dir / "auc_by_language.csv")

        _plot_budget_curve(df, plot_dir / "accuracy_vs_budget", metric_col="accuracy")
        _plot_examples_fit(df, plot_dir / "examples_fit_vs_budget")
        _plot_cost_per_correct(df, plot_dir / "cost_per_correct")

    write_text(_render_summary_md(rows, auc_rows, cfg), output_dir / "summary.md")

    return {"n_rows": len(rows), "output_dir": str(output_dir)}


# ----- plots -------------------------------------------------------------

def _plot_budget_curve(df, path, metric_col: str = "accuracy") -> None:
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots()
    for lang, sub in df.groupby("language"):
        sub = sub.sort_values("budget")
        ax.plot(sub["budget"], sub[metric_col], marker="o", label=lang)
    ax.set_xlabel("Token budget")
    ax.set_ylabel(metric_col)
    ax.set_title(f"{metric_col} vs token budget by language")
    ax.legend()
    save_figure(fig, path)


def _plot_examples_fit(df, path) -> None:
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots()
    for lang, sub in df.groupby("language"):
        sub = sub.sort_values("budget")
        ax.plot(sub["budget"], sub["examples_fit_mean"], marker="o", label=lang)
    ax.set_xlabel("Token budget")
    ax.set_ylabel("Mean demonstrations fit")
    ax.set_title("Demonstrations that fit into budget by language")
    ax.legend()
    save_figure(fig, path)


def _plot_cost_per_correct(df, path) -> None:
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots()
    sub = df.dropna(subset=["cost_per_correct"])
    if sub.empty:
        return
    for lang, g in sub.groupby("language"):
        g = g.sort_values("budget")
        ax.plot(g["budget"], g["cost_per_correct"], marker="o", label=lang)
    ax.set_xlabel("Token budget")
    ax.set_ylabel("Cost per correct (USD)")
    ax.set_title("Cost per correct prediction by language")
    ax.legend()
    save_figure(fig, path)


def _render_summary_md(rows, auc_rows, cfg) -> str:
    if not rows:
        return "# Experiment 2 — Fixed-context fairness\n\n_No rows produced._\n"
    import pandas as pd

    df = pd.DataFrame(rows).sort_values(["task", "language", "budget"])
    out = [
        "# Experiment 2 — Fixed-context fairness",
        "",
        f"Backend: `{cfg.backend.get('kind')}` / measure tokenizer: `{cfg.measure_tokenizer}`",
        "",
        "## Per-(language, budget) summary",
        "",
        df.to_markdown(index=False, floatfmt=".3f"),
        "",
    ]
    if auc_rows:
        out += [
            "## AUC of accuracy vs budget (trapezoidal, range-normalized)",
            "",
            pd.DataFrame(auc_rows).to_markdown(index=False, floatfmt=".3f"),
            "",
        ]
    out += [
        "## Reading guide",
        "",
        "- `examples_fit_mean` shows how many demonstrations the budget can hold per language",
        "  with the *measure tokenizer*. Differences here are a direct expression of",
        "  context-window economics, independent of model quality.",
        "- `accuracy` (or `f1`, `em`, `rougeL`) at each budget reflects the *combined* effect",
        "  of demonstration availability and any underlying model behaviour.",
        "- `cost_per_correct` translates the same axis into dollar terms when a hosted backend",
        "  is used.",
        "",
        "## Guardrails",
        "",
        "- Differences are **task- and tokenizer-specific**. Do not generalise to broad claims",
        "  about language ability.",
        "- For interpretation, pair these results with the Experiment 1 audit so the reader",
        "  can separate *compression* from *downstream* effects.",
    ]
    return "\n".join(out)
