"""Experiment 3: Chinese character / radical sensitivity.

We construct triples and pairs where we control:
    - shared radical vs different radical
    - shared first token (in a tokenizer's split) vs different first token

Tasks:
    - radical recognition: "Which radical does 河 contain?"
    - similarity judgment: "Are 河 and 湖 similar in meaning? Yes/No"
    - odd-one-out: "Which is the odd one out: 河, 湖, 山?"

The dataset is a small CSV the user supplies (or we ship a tiny built-in
example) with columns: char, pinyin, radical, freq_rank.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from random import Random
from typing import Any

import numpy as np
import pandas as pd

from ..backends import GenerationRequest, build_backend, warn_if_dummy
from ..plotting import apply_style, save_figure
from ..tokenizers import build_tokenizer
from ..utils import get_logger, write_csv, write_jsonl, write_text

log = get_logger(__name__)

# A minimal seed list so the experiment runs end-to-end out of the box.
# Replace by pointing config.dataset_csv at a richer file.
BUILTIN_CHARS: list[dict[str, Any]] = [
    {"char": "河", "pinyin": "hé", "radical": "氵", "freq_rank": 800},
    {"char": "湖", "pinyin": "hú", "radical": "氵", "freq_rank": 1500},
    {"char": "海", "pinyin": "hǎi", "radical": "氵", "freq_rank": 300},
    {"char": "江", "pinyin": "jiāng", "radical": "氵", "freq_rank": 600},
    {"char": "山", "pinyin": "shān", "radical": "山", "freq_rank": 200},
    {"char": "岭", "pinyin": "lǐng", "radical": "山", "freq_rank": 2000},
    {"char": "峰", "pinyin": "fēng", "radical": "山", "freq_rank": 1200},
    {"char": "树", "pinyin": "shù", "radical": "木", "freq_rank": 700},
    {"char": "林", "pinyin": "lín", "radical": "木", "freq_rank": 900},
    {"char": "森", "pinyin": "sēn", "radical": "木", "freq_rank": 1800},
    {"char": "火", "pinyin": "huǒ", "radical": "火", "freq_rank": 400},
    {"char": "灯", "pinyin": "dēng", "radical": "火", "freq_rank": 1100},
    {"char": "炉", "pinyin": "lú", "radical": "火", "freq_rank": 2500},
]


@dataclass
class RadicalConfig:
    backend: dict[str, Any] = field(default_factory=lambda: {"kind": "dummy"})
    measure_tokenizer: Any = "qwen25"
    dataset_csv: str | None = None
    n_pairs: int = 80
    n_odd_one_out: int = 40
    output_dir: str = "results/radical_sensitivity"
    seed: int = 0
    max_new_tokens: int = 16
    temperature: float = 0.0


def _load_chars(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return BUILTIN_CHARS
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Radical dataset CSV not found: {p}")
    out: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["freq_rank"] = int(row.get("freq_rank") or 0)
            out.append(row)
    return out


def _first_token_id(text: str, tok) -> int:
    ids = tok.encode(text)
    return ids[0] if ids else -1


def _build_similarity_pairs(chars: list[dict[str, Any]], n: int, rng: Random,
                            tok) -> list[dict[str, Any]]:
    """Half same-radical, half different-radical pairs, balanced when possible."""
    same = [(a, b) for a, b in combinations(chars, 2) if a["radical"] == b["radical"]]
    diff = [(a, b) for a, b in combinations(chars, 2) if a["radical"] != b["radical"]]
    rng.shuffle(same)
    rng.shuffle(diff)
    half = n // 2
    selected = same[:half] + diff[: n - half]
    rng.shuffle(selected)
    out: list[dict[str, Any]] = []
    for i, (a, b) in enumerate(selected):
        out.append({
            "id": f"sim:{i}",
            "char_a": a["char"],
            "char_b": b["char"],
            "radical_a": a["radical"],
            "radical_b": b["radical"],
            "same_radical": int(a["radical"] == b["radical"]),
            "freq_a": a["freq_rank"],
            "freq_b": b["freq_rank"],
            "first_token_a": _first_token_id(a["char"], tok),
            "first_token_b": _first_token_id(b["char"], tok),
            "same_token": int(_first_token_id(a["char"], tok) == _first_token_id(b["char"], tok)),
        })
    return out


def _build_odd_one_out(chars: list[dict[str, Any]], n: int, rng: Random) -> list[dict[str, Any]]:
    by_rad: dict[str, list[dict[str, Any]]] = {}
    for c in chars:
        by_rad.setdefault(c["radical"], []).append(c)
    rads_with_pair = [r for r, cs in by_rad.items() if len(cs) >= 2]
    other_rads = list(by_rad.keys())
    out: list[dict[str, Any]] = []
    for i in range(n):
        if not rads_with_pair:
            break
        r = rng.choice(rads_with_pair)
        a, b = rng.sample(by_rad[r], 2)
        odd_rad_choices = [x for x in other_rads if x != r and by_rad[x]]
        if not odd_rad_choices:
            continue
        c = rng.choice(by_rad[rng.choice(odd_rad_choices)])
        triple = [a, b, c]
        rng.shuffle(triple)
        odd_idx = triple.index(c)
        out.append({
            "id": f"odd:{i}",
            "chars": [t["char"] for t in triple],
            "radicals": [t["radical"] for t in triple],
            "odd_index": odd_idx,
        })
    return out


def run(cfg_dict: dict[str, Any]) -> dict[str, Any]:
    cfg = RadicalConfig(
        backend=dict(cfg_dict.get("backend", {"kind": "dummy"})),
        measure_tokenizer=cfg_dict.get("measure_tokenizer", "qwen25"),
        dataset_csv=cfg_dict.get("dataset_csv"),
        n_pairs=int(cfg_dict.get("n_pairs", 80)),
        n_odd_one_out=int(cfg_dict.get("n_odd_one_out", 40)),
        output_dir=cfg_dict.get("output_dir", "results/radical_sensitivity"),
        seed=int(cfg_dict.get("seed", 0)),
        max_new_tokens=int(cfg_dict.get("max_new_tokens", 16)),
        temperature=float(cfg_dict.get("temperature", 0.0)),
    )
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = Random(cfg.seed)
    chars = _load_chars(cfg.dataset_csv)
    backend = build_backend(cfg.backend)
    warn_if_dummy(backend, experiment="radical_sensitivity")

    try:
        tok = build_tokenizer(cfg.measure_tokenizer)
    except Exception as e:
        log.warning("Tokenizer %r unavailable (%s); using char baseline.", cfg.measure_tokenizer, e)
        tok = build_tokenizer("char")

    sim_pairs = _build_similarity_pairs(chars, cfg.n_pairs, rng, tok)
    odd_triples = _build_odd_one_out(chars, cfg.n_odd_one_out, rng)

    # ---- Similarity judgment ------------------------------------------------
    sim_rows: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    for p in sim_pairs:
        prompt = (
            f"判断下面两个汉字在意义上是否相似（同类别）。仅回答 是 或 否。\n"
            f"汉字A：{p['char_a']}\n汉字B：{p['char_b']}\n回答："
        )
        resp = backend.generate(GenerationRequest(
            prompt=prompt, max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature, seed=cfg.seed,
        ))
        ans_text = (resp.text or "").strip()
        pred_yes = int(any(s in ans_text for s in ["是", "Yes", "yes", "相似"]))
        # Treat same_radical as proxy ground truth for "similar".
        gold = p["same_radical"]
        sim_rows.append({**p, "pred_yes": pred_yes, "gold_yes": gold,
                          "correct": int(pred_yes == gold)})
        raw.append({"task": "similarity", **p, "prompt": prompt, "response": resp.text})

    # ---- Odd one out --------------------------------------------------------
    odd_rows: list[dict[str, Any]] = []
    for t in odd_triples:
        items = "、".join(t["chars"])
        prompt = (
            f"以下三个汉字中哪一个与其他两个最不同？只回答数字 1, 2, 或 3。\n"
            f"1. {t['chars'][0]}\n2. {t['chars'][1]}\n3. {t['chars'][2]}\n回答："
        )
        resp = backend.generate(GenerationRequest(
            prompt=prompt, max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature, seed=cfg.seed,
        ))
        text = (resp.text or "").strip()
        pred_idx = -1
        for i, ch in enumerate("123"):
            if ch in text:
                pred_idx = i
                break
        gold_idx = t["odd_index"]
        odd_rows.append({"id": t["id"], "chars": "/".join(t["chars"]),
                         "radicals": "/".join(t["radicals"]),
                         "pred_index": pred_idx, "gold_index": gold_idx,
                         "correct": int(pred_idx == gold_idx)})
        raw.append({"task": "odd_one_out", **t, "prompt": prompt, "response": resp.text})

    # ---- Save raw + per-condition tables ------------------------------------
    write_csv(sim_rows, out_dir / "similarity_pairs.csv")
    write_csv(odd_rows, out_dir / "odd_one_out.csv")
    write_jsonl(raw, out_dir / "raw.jsonl")

    # ---- Condition-wise accuracy --------------------------------------------
    sim_df = pd.DataFrame(sim_rows)
    cond_rows: list[dict[str, Any]] = []
    if not sim_df.empty:
        for (sr, st), sub in sim_df.groupby(["same_radical", "same_token"]):
            cond_rows.append({
                "same_radical": int(sr),
                "same_token": int(st),
                "n": int(len(sub)),
                "accuracy": float(sub["correct"].mean()),
                "p_pred_yes": float(sub["pred_yes"].mean()),
            })
    write_csv(cond_rows, out_dir / "condition_table.csv")

    # ---- Logistic regression: pred_yes ~ same_radical + same_token ---------
    # We pre-check for degenerate data (constant outcome, constant predictor)
    # because statsmodels' Logit otherwise iterates and warns once per step
    # before failing with an opaque "Singular matrix" error. The most common
    # cause is running against the dummy backend, which returns the same
    # response for every prompt and therefore yields pred_yes = constant.
    coef_rows: list[dict[str, Any]] = []
    if sim_df.empty:
        log.warning("Skipping regression: similarity table is empty.")
    else:
        n_unique_y = sim_df["pred_yes"].nunique()
        if n_unique_y < 2:
            constant_value = int(sim_df["pred_yes"].iloc[0])
            log.warning(
                "Skipping regression: pred_yes is constant (%d) across all %d "
                "rows. This usually means the backend produced uniform answers "
                "(the dummy backend always says 'OK'). Switch `backend.kind` "
                "to `hf_local` or `openai` for a meaningful regression.",
                constant_value, len(sim_df),
            )
        else:
            n_unique_sr = sim_df["same_radical"].nunique()
            n_unique_st = sim_df["same_token"].nunique()
            try:
                import warnings as _warnings

                import statsmodels.api as sm
                from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

                cols = ["same_radical", "same_token"]
                if n_unique_sr < 2:
                    cols.remove("same_radical")
                    log.info("`same_radical` is constant; dropped from regression.")
                if n_unique_st < 2:
                    cols.remove("same_token")
                    log.info("`same_token` is constant; dropped from regression.")
                if not cols:
                    raise ValueError("No non-constant predictors remain after filtering.")
                X = sm.add_constant(sim_df[cols].astype(float))
                y = sim_df["pred_yes"].astype(int)
                with _warnings.catch_warnings():
                    _warnings.simplefilter("ignore", PerfectSeparationWarning)
                    model = sm.Logit(y, X).fit(disp=False)
                for name, coef, pval in zip(
                    model.params.index, model.params.values, model.pvalues.values
                ):
                    coef_rows.append({"term": name, "coef": float(coef),
                                      "p_value": float(pval)})
            except Exception as e:
                log.warning("Skipping regression (%s).", e)
    write_csv(coef_rows, out_dir / "logit_coefficients.csv")

    if coef_rows:
        _plot_coefficients(coef_rows, out_dir / "plots" / "coefficients")

    # ---- Markdown summary ---------------------------------------------------
    md = ["# Experiment 3 — Chinese radical sensitivity", "",
          "## Condition-wise accuracy (similarity task)", ""]
    if cond_rows:
        md.append(pd.DataFrame(cond_rows).to_markdown(index=False, floatfmt=".3f"))
    else:
        md.append("_No similarity rows._")
    md += ["", "## Logistic regression coefficients", ""]
    if coef_rows:
        md.append(pd.DataFrame(coef_rows).to_markdown(index=False, floatfmt=".3f"))
    else:
        md.append("_Regression skipped._")
    md += ["", "## Error analysis notes", "",
           "- Inspect `raw.jsonl` for cases where `pred_yes=1` but `same_radical=0`",
           "  (model overgeneralises) or `pred_yes=0` and `same_radical=1` (model misses).",
           "- A non-zero `same_token` coefficient is evidence that *tokenizer* identity,",
           "  not just *script* identity, modulates the model's similarity judgment."]
    write_text("\n".join(md), out_dir / "summary.md")

    return {"n_similarity": len(sim_rows), "n_odd": len(odd_rows),
            "output_dir": str(out_dir)}


def _plot_coefficients(coef_rows: list[dict[str, Any]], path) -> None:
    import matplotlib.pyplot as plt

    apply_style()
    df = pd.DataFrame(coef_rows)
    fig, ax = plt.subplots()
    ax.barh(df["term"], df["coef"])
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel("Logit coefficient (pred_yes)")
    ax.set_title("Drivers of similarity judgment")
    save_figure(fig, path)
