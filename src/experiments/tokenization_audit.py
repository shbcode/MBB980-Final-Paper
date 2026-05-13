"""Experiment 1: tokenization audit.

For every aligned EN/ZH pair and every tokenizer, compute:
    - token count
    - chars-per-token
    - bytes-per-token
    - tokenization parity TP = zh_tokens / en_tokens
    - context shrinkage = 1 - 1/TP

Outputs:
    per_sentence.csv          one row per (paired_id, tokenizer)
    summary_by_tokenizer.csv  mean TP, CIs, Wilcoxon p-values
    plots/                    PNG + PDF (bar, box, scatter)
    summary.md                interpretation
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..data import BilingualPair, NormalizationOptions, load_bilingual
from ..plotting import apply_style, save_figure
from ..stats import bootstrap_ci, paired_wilcoxon
from ..tokenizers import TokenizerSpec, build_tokenizer
from ..utils import get_logger, write_csv, write_text

log = get_logger(__name__)


@dataclass
class AuditConfig:
    datasets: list[dict[str, Any]]
    tokenizers: list[Any]  # str presets or dicts
    languages: tuple[str, str] = ("en", "zh")
    cache_dir: str = "data/raw"
    output_dir: str = "results/tokenization_audit"
    max_examples: int | None = None
    bootstrap_n: int = 2000
    seed: int = 0
    normalization: NormalizationOptions = NormalizationOptions()


def _build_tokenizers(specs: list[Any]) -> list[TokenizerSpec]:
    out: list[TokenizerSpec] = []
    for spec in specs:
        try:
            tok = build_tokenizer(spec)
            out.append(tok)
            log.info("Loaded tokenizer: %s (family=%s, vocab=%s)",
                     tok.name, tok.family, tok.vocab_size)
        except Exception as e:
            log.warning("Skipping tokenizer %r: %s", spec, e)
    if not out:
        raise RuntimeError("No tokenizers could be loaded.")
    return out


def _row_metrics(text: str, tok: TokenizerSpec) -> dict[str, float | int]:
    ids = tok.encode(text)
    n_tokens = len(ids)
    n_chars = len(text)
    n_bytes = len(text.encode("utf-8"))
    return {
        "tokens": n_tokens,
        "chars": n_chars,
        "bytes": n_bytes,
        "chars_per_token": (n_chars / n_tokens) if n_tokens else float("nan"),
        "bytes_per_token": (n_bytes / n_tokens) if n_tokens else float("nan"),
    }


def per_sentence_table(
    pairs: list[BilingualPair],
    tokenizers: list[TokenizerSpec],
    languages: tuple[str, str],
) -> pd.DataFrame:
    """Wide table: one row per (paired_id, tokenizer), with token counts for both langs."""
    by_pair: dict[str, dict[str, BilingualPair]] = defaultdict(dict)
    for p in pairs:
        by_pair[p.paired_id][p.language] = p

    a, b = languages
    rows: list[dict[str, Any]] = []
    for pid, sides in by_pair.items():
        if a not in sides or b not in sides:
            continue  # already validated upstream, but defensive
        for tok in tokenizers:
            ma = _row_metrics(sides[a].text, tok)
            mb = _row_metrics(sides[b].text, tok)
            tp = mb["tokens"] / ma["tokens"] if ma["tokens"] else float("nan")
            shrinkage = 1 - 1 / tp if tp and not np.isnan(tp) and tp != 0 else float("nan")
            rows.append({
                "paired_id": pid,
                "dataset": sides[a].dataset,
                "split": sides[a].split,
                "domain": sides[a].domain,
                "tokenizer": tok.name,
                "tokenizer_family": tok.family,
                f"{a}_tokens": ma["tokens"],
                f"{b}_tokens": mb["tokens"],
                f"{a}_chars": ma["chars"],
                f"{b}_chars": mb["chars"],
                f"{a}_bytes": ma["bytes"],
                f"{b}_bytes": mb["bytes"],
                f"{a}_chars_per_token": ma["chars_per_token"],
                f"{b}_chars_per_token": mb["chars_per_token"],
                f"{a}_bytes_per_token": ma["bytes_per_token"],
                f"{b}_bytes_per_token": mb["bytes_per_token"],
                "TP": tp,
                "context_shrinkage": shrinkage,
            })
    return pd.DataFrame(rows)


def summary_by_tokenizer(df: pd.DataFrame, languages: tuple[str, str], n_boot: int, seed: int) -> pd.DataFrame:
    a, b = languages
    rows: list[dict[str, Any]] = []
    for (tok_name, dataset), sub in df.groupby(["tokenizer", "dataset"]):
        tp = sub["TP"].dropna().to_numpy()
        a_tok = sub[f"{a}_tokens"].to_numpy()
        b_tok = sub[f"{b}_tokens"].to_numpy()
        ci = bootstrap_ci(tp, np.mean, n_boot=n_boot, seed=seed) if tp.size else None
        wt = paired_wilcoxon(b_tok, a_tok, alternative="two-sided")
        rows.append({
            "tokenizer": tok_name,
            "dataset": dataset,
            "n_pairs": int(len(sub)),
            "mean_TP": float(np.mean(tp)) if tp.size else float("nan"),
            "median_TP": float(np.median(tp)) if tp.size else float("nan"),
            "TP_ci_low": ci.low if ci else float("nan"),
            "TP_ci_high": ci.high if ci else float("nan"),
            "mean_context_shrinkage": float(np.mean(sub["context_shrinkage"])),
            f"mean_{a}_tokens": float(np.mean(a_tok)),
            f"mean_{b}_tokens": float(np.mean(b_tok)),
            "wilcoxon_stat": wt.statistic,
            "wilcoxon_pvalue": wt.pvalue,
        })
    return pd.DataFrame(rows).sort_values(["dataset", "tokenizer"]).reset_index(drop=True)


# ----- plots -------------------------------------------------------------

def plot_mean_tp_bar(summary: pd.DataFrame, languages: tuple[str, str], path: str | Path) -> None:
    import matplotlib.pyplot as plt

    apply_style()
    a, b = languages
    fig, ax = plt.subplots()
    pivot = summary.pivot_table(index="tokenizer", columns="dataset", values="mean_TP")
    pivot = pivot.sort_values(by=pivot.columns[0])
    x = np.arange(len(pivot.index))
    width = 0.8 / max(1, len(pivot.columns))
    for i, ds in enumerate(pivot.columns):
        ax.bar(x + i * width - 0.4 + width / 2, pivot[ds].values, width=width, label=ds)
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=30, ha="right")
    ax.set_ylabel(f"Mean TP = {b}_tokens / {a}_tokens")
    ax.set_title(f"Tokenization parity ({b} vs {a}) by tokenizer")
    if len(pivot.columns) > 1:
        ax.legend(title="dataset")
    save_figure(fig, path)


def plot_tp_box(per_sent: pd.DataFrame, languages: tuple[str, str], path: str | Path) -> None:
    import matplotlib.pyplot as plt

    apply_style()
    a, b = languages
    fig, ax = plt.subplots()
    groups = per_sent.groupby("tokenizer")["TP"].apply(lambda s: s.dropna().to_numpy())
    order = sorted(groups.index, key=lambda k: float(np.median(groups[k])))
    data = [groups[k] for k in order]
    ax.boxplot(data, labels=order, showfliers=False, patch_artist=False)
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_ylabel(f"TP = {b}_tokens / {a}_tokens (per sentence)")
    ax.set_title("Per-sentence tokenization parity by tokenizer")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    save_figure(fig, path)


def plot_en_zh_scatter(per_sent: pd.DataFrame, languages: tuple[str, str], path: str | Path) -> None:
    import matplotlib.pyplot as plt

    apply_style()
    a, b = languages
    fig, ax = plt.subplots()
    tokenizers = sorted(per_sent["tokenizer"].unique())
    cmap = plt.get_cmap("tab10")
    max_v = 0.0
    for i, tk in enumerate(tokenizers):
        sub = per_sent[per_sent["tokenizer"] == tk]
        ax.scatter(sub[f"{a}_tokens"], sub[f"{b}_tokens"], s=8, alpha=0.5,
                   color=cmap(i % 10), label=tk)
        max_v = max(max_v, float(sub[f"{a}_tokens"].max()), float(sub[f"{b}_tokens"].max()))
    lim = max_v * 1.05
    ax.plot([0, lim], [0, lim], color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel(f"{a} tokens")
    ax.set_ylabel(f"{b} tokens")
    ax.set_title(f"Per-sentence token counts: {b} vs {a}")
    ax.legend(loc="upper left", fontsize=8)
    save_figure(fig, path)


# ----- markdown summary --------------------------------------------------

def render_summary_md(summary: pd.DataFrame, languages: tuple[str, str]) -> str:
    a, b = languages
    lines = [
        "# Experiment 1 — Tokenization audit",
        "",
        f"**Languages:** `{a}` vs `{b}`. **Metric:** TP = {b}_tokens / {a}_tokens.",
        "TP > 1 means the tokenizer takes more tokens to encode the target",
        "language than the source language for the same content.",
        "",
        "## Per-tokenizer summary",
        "",
        summary.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Reading guide",
        "",
        "- TP near 1.0 = symmetric efficiency between the two languages.",
        f"- High TP for `{b}` indicates the tokenizer fragments {b} more, which",
        f"  proportionally shrinks the effective context window for `{b}` users.",
        f"- *Context shrinkage* = 1 - 1/TP estimates the fraction of the",
        f"  effective context lost when switching to `{b}` at fixed token budgets.",
        "",
        "## Interpretation guardrails",
        "",
        f"- These numbers are **tokenizer-specific**. They do **not**",
        f"  themselves measure model quality or claim that `{a}` is",
        f"  intrinsically a 'better' language.",
        "- Token compression, context-window consequences, and downstream task",
        "  performance must be reported as separate axes (see Experiment 2).",
    ]
    return "\n".join(lines)


# ----- top-level driver --------------------------------------------------

def run(cfg_dict: dict[str, Any]) -> dict[str, Any]:
    """Programmatic entrypoint. `cfg_dict` matches configs/audit.yaml."""
    cfg = AuditConfig(
        datasets=cfg_dict["datasets"],
        tokenizers=cfg_dict["tokenizers"],
        languages=tuple(cfg_dict.get("languages", ["en", "zh"])),  # type: ignore[arg-type]
        cache_dir=cfg_dict.get("cache_dir", "data/raw"),
        output_dir=cfg_dict.get("output_dir", "results/tokenization_audit"),
        max_examples=cfg_dict.get("max_examples"),
        bootstrap_n=cfg_dict.get("bootstrap_n", 2000),
        seed=cfg_dict.get("seed", 0),
        normalization=NormalizationOptions(**cfg_dict.get("normalization", {})),
    )
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"

    tokenizers = _build_tokenizers(cfg.tokenizers)

    all_pairs: list[BilingualPair] = []
    for ds_spec in cfg.datasets:
        pairs = load_bilingual(
            dataset=ds_spec["name"],
            split=ds_spec.get("split"),
            languages=cfg.languages,
            cache_dir=cfg.cache_dir,
            max_examples=ds_spec.get("max_examples", cfg.max_examples),
            normalization=cfg.normalization,
        )
        all_pairs.extend(pairs)

    log.info("Loaded %d total bilingual rows across %d dataset(s)",
             len(all_pairs), len(cfg.datasets))

    per_sent = per_sentence_table(all_pairs, tokenizers, cfg.languages)
    summary = summary_by_tokenizer(per_sent, cfg.languages, cfg.bootstrap_n, cfg.seed)

    write_csv(per_sent.to_dict(orient="records"), output_dir / "per_sentence.csv")
    write_csv(summary.to_dict(orient="records"), output_dir / "summary_by_tokenizer.csv")
    write_text(render_summary_md(summary, cfg.languages), output_dir / "summary.md")

    plot_mean_tp_bar(summary, cfg.languages, plot_dir / "tp_bar")
    plot_tp_box(per_sent, cfg.languages, plot_dir / "tp_box")
    plot_en_zh_scatter(per_sent, cfg.languages, plot_dir / "en_zh_scatter")

    return {
        "n_pairs": int(per_sent["paired_id"].nunique()),
        "n_tokenizers": len(tokenizers),
        "n_rows_per_sentence_csv": int(len(per_sent)),
        "output_dir": str(output_dir),
    }
