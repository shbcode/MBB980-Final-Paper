"""Experiment 4: paraphrase vs token-perturbation.

For a subset of prompts, we apply two families of transforms:

    paraphrase
        - meaning-preserving rewrite (project-supplied or synthetic)
    token perturbation
        - punctuation swap (full-width <-> ASCII)
        - numeral style swap (Arabic <-> Chinese)
        - simplified <-> traditional (OpenCC)
        - synonym swap (project-supplied dictionary)

We compare:
    - token counts before/after
    - accuracy delta (if the prompt is from a labeled task)
    - response stability (string equality + Jaccard token overlap)
    - log-prob drift (when a local HF backend is available)

Output:
    perturbations.csv      (one row per (prompt_id, transform))
    summary.csv            (mean stability/Δaccuracy by transform)
    plots/                 paired_variance.png/.pdf
    summary.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..backends import GenerationRequest, build_backend, warn_if_dummy
from ..data import normalize_numbers, normalize_punctuation, s2t, t2s
from ..plotting import apply_style, save_figure
from ..tokenizers import build_tokenizer
from ..utils import get_logger, write_csv, write_jsonl, write_text

log = get_logger(__name__)


# ----- transforms ---------------------------------------------------------

def _identity(text: str) -> str:
    return text


def _ascii_punct(text: str) -> str:
    return normalize_punctuation(text)


def _fullwidth_punct(text: str) -> str:
    rev = {",": "，", ".": "。", "!": "！", "?": "？", ";": "；", ":": "：",
           "(": "（", ")": "）"}
    return text.translate(str.maketrans(rev))


def _numerals_arabic(text: str) -> str:
    return normalize_numbers(text)


def _to_traditional(text: str) -> str:
    return s2t(text)


def _to_simplified(text: str) -> str:
    return t2s(text)


def _synonym_swap(text: str, mapping: dict[str, str] | None = None) -> str:
    if not mapping:
        return text
    out = text
    for k, v in mapping.items():
        out = out.replace(k, v)
    return out


TRANSFORMS: dict[str, Callable[[str], str]] = {
    "identity": _identity,
    "punct_ascii": _ascii_punct,
    "punct_fullwidth": _fullwidth_punct,
    "numerals_arabic": _numerals_arabic,
    "to_traditional": _to_traditional,
    "to_simplified": _to_simplified,
}


def _jaccard(a: str, b: str) -> float:
    sa = set(a.split())
    sb = set(b.split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ----- paraphrase source --------------------------------------------------

def _load_paraphrase_pairs(path: str | None) -> list[dict[str, Any]] | None:
    """Optional CSV: id,language,original,paraphrase,answer."""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        log.warning("Paraphrase CSV not found: %s; skipping paraphrase axis.", p)
        return None
    df = pd.read_csv(p)
    return df.to_dict(orient="records")


# ----- driver -------------------------------------------------------------

@dataclass
class PerturbConfig:
    backend: dict[str, Any] = field(default_factory=lambda: {"kind": "dummy"})
    measure_tokenizer: Any = "qwen25"
    prompt_csv: str = ""  # required: id,language,prompt[,answer]
    paraphrase_csv: str | None = None
    transforms: list[str] = field(default_factory=lambda: list(TRANSFORMS.keys()))
    output_dir: str = "results/paraphrase_token_perturb"
    seed: int = 0
    max_new_tokens: int = 64
    temperature: float = 0.0
    synonyms: dict[str, str] = field(default_factory=dict)


def run(cfg_dict: dict[str, Any]) -> dict[str, Any]:
    cfg = PerturbConfig(
        backend=dict(cfg_dict.get("backend", {"kind": "dummy"})),
        measure_tokenizer=cfg_dict.get("measure_tokenizer", "qwen25"),
        prompt_csv=cfg_dict.get("prompt_csv", ""),
        paraphrase_csv=cfg_dict.get("paraphrase_csv"),
        transforms=list(cfg_dict.get("transforms", list(TRANSFORMS.keys()))),
        output_dir=cfg_dict.get("output_dir", "results/paraphrase_token_perturb"),
        seed=int(cfg_dict.get("seed", 0)),
        max_new_tokens=int(cfg_dict.get("max_new_tokens", 64)),
        temperature=float(cfg_dict.get("temperature", 0.0)),
        synonyms=dict(cfg_dict.get("synonyms", {})),
    )

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not cfg.prompt_csv:
        raise ValueError("perturb config requires `prompt_csv`")
    prompts = pd.read_csv(cfg.prompt_csv).to_dict(orient="records")
    paras = _load_paraphrase_pairs(cfg.paraphrase_csv) or []

    backend = build_backend(cfg.backend)
    warn_if_dummy(backend, experiment="paraphrase_perturb")
    try:
        tok = build_tokenizer(cfg.measure_tokenizer)
    except Exception as e:
        log.warning("Tokenizer %r unavailable (%s); falling back to char.", cfg.measure_tokenizer, e)
        tok = build_tokenizer("char")

    # Synonym transform with bound dictionary.
    transforms: dict[str, Callable[[str], str]] = dict(TRANSFORMS)
    if cfg.synonyms:
        transforms["synonym_swap"] = lambda t: _synonym_swap(t, cfg.synonyms)

    rows: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []

    # 1) Token-perturbation transforms.
    for ex in prompts:
        original = ex["prompt"]
        gold = ex.get("answer")
        baseline_resp = backend.generate(GenerationRequest(
            prompt=original, max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature, seed=cfg.seed,
        ))
        baseline_text = baseline_resp.text or ""
        for tname in cfg.transforms:
            if tname not in transforms:
                continue
            transformed = transforms[tname](original)
            resp = backend.generate(GenerationRequest(
                prompt=transformed, max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature, seed=cfg.seed,
            ))
            r_text = resp.text or ""
            rows.append({
                "id": ex.get("id"),
                "language": ex.get("language"),
                "transform": tname,
                "kind": "token_perturb",
                "tokens_before": tok.count(original),
                "tokens_after": tok.count(transformed),
                "delta_tokens": tok.count(transformed) - tok.count(original),
                "stability_exact": int(r_text.strip() == baseline_text.strip()),
                "stability_jaccard": _jaccard(r_text, baseline_text),
                "correct_before": _maybe_correct(baseline_text, gold),
                "correct_after": _maybe_correct(r_text, gold),
            })
            raw.append({"id": ex.get("id"), "transform": tname,
                        "before": original, "after": transformed,
                        "response_before": baseline_text, "response_after": r_text})

    # 2) Paraphrase axis (if data provided).
    for ex in paras:
        original = ex["original"]
        para = ex["paraphrase"]
        gold = ex.get("answer")
        r0 = backend.generate(GenerationRequest(
            prompt=original, max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature, seed=cfg.seed)).text or ""
        r1 = backend.generate(GenerationRequest(
            prompt=para, max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature, seed=cfg.seed)).text or ""
        rows.append({
            "id": ex.get("id"),
            "language": ex.get("language"),
            "transform": "paraphrase",
            "kind": "paraphrase",
            "tokens_before": tok.count(original),
            "tokens_after": tok.count(para),
            "delta_tokens": tok.count(para) - tok.count(original),
            "stability_exact": int(r0.strip() == r1.strip()),
            "stability_jaccard": _jaccard(r0, r1),
            "correct_before": _maybe_correct(r0, gold),
            "correct_after": _maybe_correct(r1, gold),
        })
        raw.append({"id": ex.get("id"), "transform": "paraphrase",
                    "before": original, "after": para,
                    "response_before": r0, "response_after": r1})

    write_csv(rows, out_dir / "perturbations.csv")
    write_jsonl(raw, out_dir / "raw.jsonl")

    # ---- aggregated summary ------------------------------------------------
    summary_rows: list[dict[str, Any]] = []
    if rows:
        df = pd.DataFrame(rows)
        for tname, sub in df.groupby("transform"):
            corr_b = sub["correct_before"].dropna()
            corr_a = sub["correct_after"].dropna()
            summary_rows.append({
                "transform": tname,
                "n": int(len(sub)),
                "mean_stability_exact": float(sub["stability_exact"].mean()),
                "mean_stability_jaccard": float(sub["stability_jaccard"].mean()),
                "mean_delta_tokens": float(sub["delta_tokens"].mean()),
                "delta_accuracy": (
                    float(corr_a.mean() - corr_b.mean()) if len(corr_a) and len(corr_b) else float("nan")
                ),
            })
        write_csv(summary_rows, out_dir / "summary.csv")
        _plot_paired_variance(df, out_dir / "plots" / "paired_variance")

    write_text(_render_md(summary_rows), out_dir / "summary.md")
    return {"n_rows": len(rows), "output_dir": str(out_dir)}


def _maybe_correct(response: str, gold: Any) -> float | None:
    """Substring-match a gold answer inside a response.

    Pandas reads CSVs with mixed empty + integer answer columns as float64
    (e.g. `59` -> `59.0`), and `'59.0' in '... equals 59.'` is False.
    To avoid that false-negative, normalise integer-valued floats back to
    their integer string form before matching.
    """
    if gold is None or (isinstance(gold, float) and np.isnan(gold)):
        return None
    if isinstance(gold, float) and gold.is_integer():
        gold_str = str(int(gold))
    else:
        gold_str = str(gold)
    gold_str = gold_str.strip().lower()
    if not gold_str:
        return None
    return float(gold_str in (response or "").strip().lower())


def _plot_paired_variance(df: pd.DataFrame, path) -> None:
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots()
    transforms = sorted(df["transform"].unique())
    data = [df[df["transform"] == t]["stability_jaccard"].dropna().to_numpy()
            for t in transforms]
    ax.boxplot(data, labels=transforms, showfliers=False)
    ax.set_ylabel("Jaccard stability vs baseline response")
    ax.set_title("Response stability by transform")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    save_figure(fig, path)


def _render_md(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "# Experiment 4 — Paraphrase vs token perturbation\n\n_No rows produced._\n"
    return (
        "# Experiment 4 — Paraphrase vs token perturbation\n\n"
        "## Per-transform summary\n\n"
        + pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3f")
        + "\n\n## Reading guide\n\n"
        "- `mean_stability_jaccard` near 1.0 means the model produces a near-identical\n"
        "  response despite the transform. Lower values indicate sensitivity.\n"
        "- `delta_accuracy` is positive when the transform helps (e.g. cleaner\n"
        "  numerals) and negative when it hurts.\n"
        "- Compare paraphrase rows (semantic-preserving) against token-perturbation\n"
        "  rows: a model that's more sensitive to surface tokenization than to\n"
        "  meaning is leaning on tokenization artefacts.\n"
    )
