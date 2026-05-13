"""Experiment 5: tokenizer adaptation.

Two modes:

A) lightweight
    - Mine the most frequent {1000, 5000, 10000} ZH n-grams missing from the
      base tokenizer and add them as new tokens.
    - Resize embeddings (mean-init the new rows).
    - Continue pretraining on a small mixed EN/ZH corpus for K steps.
    - Re-run the Experiment 1 audit and a tiny eval.

B) heavy
    - Train small matched decoder-only models from scratch with
      different tokenizers (BPE / Unigram / byte). Outside the scope of a
      single laptop run; we provide the launch script + config + integration
      test, but the actual training is left to the user.

This module focuses on the *orchestration* layer. The heavy training inside
mode A is gated behind `--do_train`; without it we still write the new
tokenizer + a before/after audit comparison for the *added vocabulary*
counterfactual ("what would tokenization look like if these tokens existed?").
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ..data import load_bilingual
from ..plotting import apply_style, save_figure
from ..tokenizers import build_tokenizer
from ..utils import get_logger, write_csv, write_text

log = get_logger(__name__)


@dataclass
class AdaptConfig:
    mode: str = "lightweight"  # "lightweight" | "heavy"
    base_tokenizer: Any = "qwen25"
    adaptation_corpus: list[dict[str, Any]] = field(default_factory=lambda: [
        {"name": "flores200", "split": "devtest"},
    ])
    new_token_counts: list[int] = field(default_factory=lambda: [1000, 5000, 10000])
    max_examples: int | None = None
    output_dir: str = "results/tokenizer_adaptation"
    seed: int = 0
    do_train: bool = False
    train_steps: int = 200
    train_lr: float = 1e-5
    train_batch_size: int = 4
    eval_audit_dataset: dict[str, Any] = field(default_factory=lambda: {
        "name": "flores200", "split": "devtest", "max_examples": 200,
    })


# ---- Vocabulary mining --------------------------------------------------

def _candidate_zh_ngrams(texts: list[str], n_max: int = 4, min_count: int = 5) -> Counter:
    """Count Han-only character n-grams up to length n_max."""
    counter: Counter = Counter()
    for t in texts:
        han = "".join(ch if "\u4e00" <= ch <= "\u9fff" else " " for ch in t)
        for chunk in han.split():
            for n in range(1, n_max + 1):
                for i in range(len(chunk) - n + 1):
                    counter[chunk[i : i + n]] += 1
    return Counter({k: v for k, v in counter.items() if v >= min_count})


def _select_new_tokens(candidates: Counter, k: int, base_tok) -> list[str]:
    """Pick top-k candidates that the base tokenizer fragments (>1 token)."""
    out: list[str] = []
    for piece, _ in candidates.most_common():
        if len(out) >= k:
            break
        if base_tok.count(piece) > 1:
            out.append(piece)
    return out


# ---- Counterfactual audit -----------------------------------------------

def _measure_with_extra_tokens(
    texts: list[str], base_tok, extra_tokens: set[str]
) -> list[int]:
    """Approximate token count after adding `extra_tokens` to the base tokenizer.

    We greedily replace longest matching extras with single tokens, then count
    base-tokenizer tokens for the remaining text. This is a *fast* proxy for
    embedding extension; the actual extension at training time may differ.
    """
    if not extra_tokens:
        return [base_tok.count(t) for t in texts]
    sorted_extras = sorted(extra_tokens, key=len, reverse=True)
    counts: list[int] = []
    for text in texts:
        merged = 0
        remaining = text
        for piece in sorted_extras:
            if not remaining:
                break
            n_hits = remaining.count(piece)
            if n_hits:
                merged += n_hits
                remaining = remaining.replace(piece, " ")  # space separates fragments
        counts.append(merged + base_tok.count(remaining))
    return counts


# ---- Heavy mode (skeleton) ----------------------------------------------

def _heavy_mode(cfg: AdaptConfig) -> dict[str, Any]:
    """Stub: we provide the structure so users can plug in their training code."""
    log.warning("Heavy mode is a skeleton; returning a launch plan, not running training.")
    plan = {
        "mode": "heavy",
        "note": (
            "Train small matched decoder-only models from scratch with different "
            "tokenizers (e.g. BPE/Unigram/byte). Recommended pathway: nanoGPT or "
            "litgpt with shared model code and three SentencePiece-trained tokenizers. "
            "Use src/tokenizers/train_sentencepiece.py to produce the tokenizers, "
            "then a separate training script outside this pipeline."
        ),
        "tokenizers_to_train": [
            {"kind": "sentencepiece", "model_type": "bpe", "vocab_size": 32000},
            {"kind": "sentencepiece", "model_type": "unigram", "vocab_size": 32000},
            {"kind": "byte"},
        ],
        "next_steps": [
            "Run train_sentencepiece for each variant.",
            "Train matched models (same arch, same data, same steps).",
            "Re-run Experiment 1 + a downstream task; record curves under "
            "results/tokenizer_adaptation/heavy/.",
        ],
    }
    out_dir = Path(cfg.output_dir) / "heavy"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    return {"mode": "heavy", "wrote": str(out_dir / "plan.json")}


# ---- Lightweight mode ---------------------------------------------------

def _lightweight_mode(cfg: AdaptConfig) -> dict[str, Any]:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_tok = build_tokenizer(cfg.base_tokenizer)

    # Gather adaptation corpus (zh side).
    zh_texts: list[str] = []
    for ds in cfg.adaptation_corpus:
        pairs = load_bilingual(
            dataset=ds["name"], split=ds.get("split"),
            max_examples=ds.get("max_examples", cfg.max_examples),
        )
        zh_texts.extend(p.text for p in pairs if p.language == "zh")
    log.info("Adaptation corpus: %d ZH sentences", len(zh_texts))

    # Eval set (separate slice).
    eval_pairs = load_bilingual(
        dataset=cfg.eval_audit_dataset["name"],
        split=cfg.eval_audit_dataset.get("split"),
        max_examples=cfg.eval_audit_dataset.get("max_examples"),
    )
    en_eval = [p.text for p in eval_pairs if p.language == "en"]
    zh_eval = [p.text for p in eval_pairs if p.language == "zh"]

    candidates = _candidate_zh_ngrams(zh_texts, n_max=4, min_count=5)
    log.info("Mined %d candidate Han n-grams (min_count=5)", len(candidates))

    # Baseline audit.
    base_en = [base_tok.count(t) for t in en_eval]
    base_zh = [base_tok.count(t) for t in zh_eval]
    base_tp = sum(z / max(1, e) for z, e in zip(base_zh, base_en)) / max(1, len(en_eval))

    rows: list[dict[str, Any]] = [{
        "added_tokens": 0,
        "mean_en_tokens": _safe_mean(base_en),
        "mean_zh_tokens": _safe_mean(base_zh),
        "mean_TP": base_tp,
    }]

    # Counterfactual audits per added-token budget.
    extras_by_k: dict[int, list[str]] = {}
    for k in cfg.new_token_counts:
        extras = _select_new_tokens(candidates, k, base_tok)
        extras_by_k[k] = extras
        log.info("k=%d -> %d actually-added tokens", k, len(extras))

        new_zh = _measure_with_extra_tokens(zh_eval, base_tok, set(extras))
        new_tp = sum(nz / max(1, e) for nz, e in zip(new_zh, base_en)) / max(1, len(en_eval))
        rows.append({
            "added_tokens": len(extras),
            "mean_en_tokens": _safe_mean(base_en),
            "mean_zh_tokens": _safe_mean(new_zh),
            "mean_TP": new_tp,
        })

        # Save the extra vocab list per budget for downstream training reuse.
        (out_dir / f"extras_k{k}.txt").write_text(
            "\n".join(extras), encoding="utf-8"
        )

    write_csv(rows, out_dir / "audit_before_after.csv")
    _plot_before_after(rows, out_dir / "plots" / "audit_before_after")
    write_text(_render_md(rows, cfg, do_train=cfg.do_train), out_dir / "summary.md")

    # Optional actual continued pretraining hook.
    if cfg.do_train:
        try:
            _continue_pretraining(cfg, extras_by_k[max(extras_by_k)], zh_texts, out_dir)
        except Exception as e:
            log.error("Continued pretraining failed: %s", e)

    return {"mode": "lightweight", "n_rows": len(rows), "output_dir": str(out_dir)}


def _safe_mean(xs: list[int]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def _continue_pretraining(
    cfg: AdaptConfig, extras: list[str], zh_texts: list[str], out_dir: Path
) -> None:
    """Load the base HF tokenizer/model, add `extras`, resize embeddings, and
    run a short LM finetune on the adaptation corpus. Optional and heavyweight.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    base_spec = cfg.base_tokenizer
    if isinstance(base_spec, str):
        # Map preset name to HF id
        from ..tokenizers.registry import _PRESETS

        if base_spec not in _PRESETS or _PRESETS[base_spec].get("kind") != "hf":
            raise ValueError("Continued pretraining requires an HF base tokenizer.")
        hf_id = _PRESETS[base_spec]["hf_id"]
    else:
        if base_spec.get("kind") != "hf":
            raise ValueError("Continued pretraining requires an HF base tokenizer.")
        hf_id = base_spec["hf_id"]

    log.info("Loading %s for continued pretraining", hf_id)
    tok = AutoTokenizer.from_pretrained(hf_id)
    n_added = tok.add_tokens(extras)
    log.info("Added %d new tokens (out of %d candidates)", n_added, len(extras))

    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.bfloat16,
                                                  device_map="auto")
    model.resize_token_embeddings(len(tok))
    model.train()

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.train_lr)

    losses: list[float] = []
    bs = cfg.train_batch_size
    for step in range(cfg.train_steps):
        batch = zh_texts[(step * bs) % len(zh_texts) : (step * bs) % len(zh_texts) + bs]
        enc = tok(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
        enc = {k: v.to(model.device) for k, v in enc.items()}
        out = model(**enc, labels=enc["input_ids"])
        loss = out.loss
        loss.backward()
        optim.step()
        optim.zero_grad()
        losses.append(float(loss.item()))
        if step % 10 == 0:
            log.info("step %d  loss=%.4f", step, loss.item())

    # Save curve + adapted artifacts.
    (out_dir / "training_loss.csv").write_text(
        "step,loss\n" + "\n".join(f"{i},{l}" for i, l in enumerate(losses)),
        encoding="utf-8",
    )
    save_dir = out_dir / "adapted_model"
    save_dir.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(save_dir)
    model.save_pretrained(save_dir)
    log.info("Saved adapted model to %s", save_dir)


def _plot_before_after(rows: list[dict[str, Any]], path) -> None:
    import matplotlib.pyplot as plt

    apply_style()
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots()
    ax.plot(df["added_tokens"], df["mean_TP"], marker="o")
    ax.axhline(1.0, color="black", linestyle="--", alpha=0.6, linewidth=0.8)
    ax.set_xlabel("# added Chinese tokens")
    ax.set_ylabel("Mean TP (zh/en)")
    ax.set_title("Tokenization parity vs added Chinese vocabulary")
    save_figure(fig, path)


def _render_md(rows, cfg, *, do_train: bool) -> str:
    return (
        "# Experiment 5 — Tokenizer adaptation (lightweight)\n\n"
        f"Base tokenizer: `{cfg.base_tokenizer}`\n\n"
        "## Counterfactual audit (added vocabulary, no retraining)\n\n"
        + pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3f")
        + "\n\n## Reading guide\n\n"
        "- The `0` row is the unmodified base tokenizer's audit.\n"
        "- Each subsequent row simulates *if* the new vocabulary existed: it\n"
        "  reports the lower-bound effect on zh token counts (greedy longest match).\n"
        "- A real trained extension typically lands between this counterfactual and\n"
        "  the base, depending on training quality.\n\n"
        + ("## Continued pretraining\n\n"
           f"Trained for `{cfg.train_steps}` steps at lr={cfg.train_lr}, batch={cfg.train_batch_size}.\n"
           "See `training_loss.csv` and `adapted_model/`.\n"
           if do_train else
           "## Continued pretraining\n\nSkipped (`do_train: false`).\n")
    )


# ---- Top-level driver ---------------------------------------------------

def run(cfg_dict: dict[str, Any]) -> dict[str, Any]:
    cfg = AdaptConfig(
        mode=cfg_dict.get("mode", "lightweight"),
        base_tokenizer=cfg_dict.get("base_tokenizer", "qwen25"),
        adaptation_corpus=list(cfg_dict.get("adaptation_corpus", [
            {"name": "flores200", "split": "devtest"}
        ])),
        new_token_counts=list(cfg_dict.get("new_token_counts", [1000, 5000, 10000])),
        max_examples=cfg_dict.get("max_examples"),
        output_dir=cfg_dict.get("output_dir", "results/tokenizer_adaptation"),
        seed=int(cfg_dict.get("seed", 0)),
        do_train=bool(cfg_dict.get("do_train", False)),
        train_steps=int(cfg_dict.get("train_steps", 200)),
        train_lr=float(cfg_dict.get("train_lr", 1e-5)),
        train_batch_size=int(cfg_dict.get("train_batch_size", 4)),
        eval_audit_dataset=dict(cfg_dict.get("eval_audit_dataset",
                                            {"name": "flores200", "split": "devtest",
                                             "max_examples": 200})),
    )
    if cfg.mode == "heavy":
        return _heavy_mode(cfg)
    return _lightweight_mode(cfg)
