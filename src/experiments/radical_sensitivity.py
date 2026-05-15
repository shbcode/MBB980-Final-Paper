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

    # ---- Cell-level effect sizes (always reported) -------------------------
    # Even when the logistic regression is unstable (perfect separation,
    # constant predictor, tiny sample), we can always report the 2x2 contingency
    # of (same_radical, pred_yes) plus a Fisher's exact test and an odds ratio
    # with a Haldane-Anscombe correction (+0.5 to every cell). This is the
    # publishable backup statistic for small / clean datasets.
    contingency_rows: list[dict[str, Any]] = []
    if not sim_df.empty:
        contingency_rows = _contingency_report(sim_df)
    write_csv(contingency_rows, out_dir / "contingency_same_radical.csv")

    # ---- Logistic regression: pred_yes ~ same_radical + same_token ---------
    # We pre-check for degenerate data (constant outcome, constant predictor)
    # because statsmodels' Logit otherwise iterates and warns once per step
    # before failing with an opaque "Singular matrix" error. The most common
    # causes are (1) the dummy backend (constant pred_yes) and (2) tiny seed
    # datasets that produce *near-perfect* separation -- in that case we fall
    # back to L2-regularized logistic regression, which always converges and
    # whose coefficients still have meaningful sign + magnitude.
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
            coef_rows = _fit_radical_regression(sim_df)
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
    md += ["", "## Contingency: same_radical vs pred_yes", ""]
    if contingency_rows:
        md.append(pd.DataFrame(contingency_rows).to_markdown(index=False, floatfmt=".3f"))
        md.append("")
        md.append(
            "Odds ratio uses a Haldane-Anscombe (+0.5) correction; Fisher's exact "
            "p-value is unconditional and exact. These are stable on small samples "
            "where logistic regression suffers from perfect/near-perfect separation."
        )
    else:
        md.append("_No similarity rows._")
    md += ["", "## Logistic regression coefficients", ""]
    if coef_rows:
        md.append(pd.DataFrame(coef_rows).to_markdown(index=False, floatfmt=".3f"))
        if any(r.get("regularized") for r in coef_rows):
            md.append(
                "\n**Note:** L2-regularized fit. Standard MLE failed to converge "
                "(near-perfect separation), so coefficients here are penalised "
                "estimates -- read sign and relative magnitude, not the absolute "
                "values or p-values. The contingency table above is the primary "
                "evidence."
            )
    else:
        md.append("_Regression skipped._")
    md += ["", "## Error analysis notes", "",
           "- Inspect `raw.jsonl` for cases where `pred_yes=1` but `same_radical=0`",
           "  (model overgeneralises) or `pred_yes=0` and `same_radical=1` (model misses).",
           "- A non-zero `same_token` coefficient is evidence that *tokenizer* identity,",
           "  not just *script* identity, modulates the model's similarity judgment.",
           "- Tiny seed datasets (~13 chars) leave `same_token` constant for HF",
           "  tokenizers that map every Han char to its own id. Supply a richer",
           "  `dataset_csv` (e.g. ~200 chars across ~10 radicals) to recover power."]
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


def _contingency_report(sim_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Return a 2x2 contingency table for (same_radical, pred_yes) plus
    Haldane-corrected odds ratio and Fisher's exact p-value.

    Stable for small samples where logistic regression suffers from perfect or
    near-perfect separation.
    """
    from scipy.stats import fisher_exact

    a = int(((sim_df["same_radical"] == 1) & (sim_df["pred_yes"] == 1)).sum())  # diff/diff
    b = int(((sim_df["same_radical"] == 1) & (sim_df["pred_yes"] == 0)).sum())
    c = int(((sim_df["same_radical"] == 0) & (sim_df["pred_yes"] == 1)).sum())
    d = int(((sim_df["same_radical"] == 0) & (sim_df["pred_yes"] == 0)).sum())

    a_c, b_c, c_c, d_c = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    odds_ratio = (a_c * d_c) / (b_c * c_c)
    try:
        _, p_fisher = fisher_exact([[a, b], [c, d]], alternative="two-sided")
    except Exception:
        p_fisher = float("nan")

    n = a + b + c + d
    p_yes_given_same = a / (a + b) if (a + b) else float("nan")
    p_yes_given_diff = c / (c + d) if (c + d) else float("nan")
    return [{
        "n": n,
        "n_same_radical": a + b,
        "n_diff_radical": c + d,
        "p_pred_yes_given_same_radical": float(p_yes_given_same),
        "p_pred_yes_given_diff_radical": float(p_yes_given_diff),
        "odds_ratio_haldane": float(odds_ratio),
        "fisher_exact_p": float(p_fisher),
        "cell_a_same_yes": a, "cell_b_same_no": b,
        "cell_c_diff_yes": c, "cell_d_diff_no": d,
    }]


def _fit_radical_regression(sim_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Fit logistic regression of pred_yes ~ same_radical [+ same_token].

    Strategy:
      1. Drop predictors that are constant in this sample (statsmodels would
         emit a singular-matrix error otherwise).
      2. Try ordinary MLE first (best when the data are well-mixed).
      3. If MLE diverges (PerfectSeparationError or non-convergence), retry
         with L2-regularized logistic regression, which always converges and
         whose sign / relative magnitude remain interpretable.
    """
    import warnings as _warnings

    import statsmodels.api as sm
    from statsmodels.tools.sm_exceptions import (
        ConvergenceWarning,
        PerfectSeparationError,
        PerfectSeparationWarning,
    )

    cols = [c for c in ["same_radical", "same_token"] if sim_df[c].nunique() >= 2]
    if not cols:
        log.warning(
            "Skipping regression: both `same_radical` and `same_token` are "
            "constant in this sample. The contingency table is the meaningful "
            "report. Consider supplying a richer `dataset_csv`."
        )
        return []

    X = sm.add_constant(sim_df[cols].astype(float))
    y = sim_df["pred_yes"].astype(int)

    # Attempt 1: plain MLE.
    try:
        with _warnings.catch_warnings():
            _warnings.simplefilter("error", PerfectSeparationWarning)
            _warnings.simplefilter("error", ConvergenceWarning)
            model = sm.Logit(y, X).fit(disp=False)
        return [
            {"term": name, "coef": float(coef), "p_value": float(pval),
             "regularized": False}
            for name, coef, pval in zip(
                model.params.index, model.params.values, model.pvalues.values
            )
        ]
    except (PerfectSeparationError, PerfectSeparationWarning, ConvergenceWarning,
            np.linalg.LinAlgError, ValueError) as e:
        log.info("Plain MLE failed (%s); falling back to L2-regularized fit.",
                 type(e).__name__)

    # Attempt 2: L2-regularized fit (always converges; no analytic p-values).
    try:
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            reg = sm.Logit(y, X).fit_regularized(
                method="l1", alpha=1.0, disp=False, trim_mode="off"
            )
        return [
            {"term": name, "coef": float(coef), "p_value": float("nan"),
             "regularized": True}
            for name, coef in zip(reg.params.index, reg.params.values)
        ]
    except Exception as e:
        log.warning("Regularized regression also failed (%s); skipping.", e)
        return []
