"""Reconstruct the entire analysis from cached CSV outputs.

This script does NOT re-run any experiment. It assumes you have already run
each experiment at least once, so the relevant CSVs exist under
`results/<experiment>/`. It then:

    1. Reloads every per-experiment summary.
    2. Rebuilds the headline plots into `results/<experiment>/plots/`.
    3. Regenerates `reports/final_report.md` from the cached numbers.

Use it after pulling someone else's `results/` archive to verify the figures
and report match the cached data.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import pandas as pd

from .experiments.fixed_context import (
    _plot_budget_curve,
    _plot_cost_per_correct,
    _plot_examples_fit,
)
from .experiments.tokenization_audit import (
    plot_en_zh_scatter,
    plot_mean_tp_bar,
    plot_tp_box,
)
from .experiments.tokenizer_adaptation import _plot_before_after
from .plotting import apply_style
from .utils import write_text


def _maybe_read(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    if path.stat().st_size == 0:
        return None
    return pd.read_csv(path)


@click.command()
@click.option("--results-dir", default="results", type=click.Path())
@click.option("--report-out", default="reports/final_report.md", type=click.Path())
def main(results_dir: str, report_out: str) -> None:
    apply_style()
    root = Path(results_dir)

    sections: list[str] = ["# Final report (reconstructed from cache)\n"]

    # ---- Experiment 1 ---------------------------------------------------
    audit_dir = root / "tokenization_audit"
    per_sent = _maybe_read(audit_dir / "per_sentence.csv")
    summary = _maybe_read(audit_dir / "summary_by_tokenizer.csv")
    if per_sent is not None and summary is not None:
        plot_dir = audit_dir / "plots"
        plot_mean_tp_bar(summary, ("en", "zh"), plot_dir / "tp_bar")
        plot_tp_box(per_sent, ("en", "zh"), plot_dir / "tp_box")
        plot_en_zh_scatter(per_sent, ("en", "zh"), plot_dir / "en_zh_scatter")
        sections.append("## Experiment 1 — Tokenization audit\n")
        sections.append(summary.to_markdown(index=False, floatfmt=".3f"))
        sections.append("")
    else:
        sections.append("## Experiment 1 — Tokenization audit\n\n_No cached data found._\n")

    # ---- Experiment 2 ---------------------------------------------------
    fc_summary = _maybe_read(root / "fixed_context" / "summary.csv")
    if fc_summary is not None and not fc_summary.empty:
        plot_dir = root / "fixed_context" / "plots"
        _plot_budget_curve(fc_summary, plot_dir / "accuracy_vs_budget", "accuracy")
        _plot_examples_fit(fc_summary, plot_dir / "examples_fit_vs_budget")
        _plot_cost_per_correct(fc_summary, plot_dir / "cost_per_correct")
        sections.append("## Experiment 2 — Fixed-context fairness\n")
        sections.append(fc_summary.to_markdown(index=False, floatfmt=".3f"))
        sections.append("")
    else:
        sections.append("## Experiment 2 — Fixed-context fairness\n\n_No cached data found._\n")

    # ---- Experiment 3 ---------------------------------------------------
    rad_cond = _maybe_read(root / "radical_sensitivity" / "condition_table.csv")
    rad_coef = _maybe_read(root / "radical_sensitivity" / "logit_coefficients.csv")
    if rad_cond is not None or rad_coef is not None:
        sections.append("## Experiment 3 — Chinese radical sensitivity\n")
        if rad_cond is not None and not rad_cond.empty:
            sections.append("### Condition-wise accuracy\n")
            sections.append(rad_cond.to_markdown(index=False, floatfmt=".3f"))
            sections.append("")
        if rad_coef is not None and not rad_coef.empty:
            sections.append("### Logit coefficients\n")
            sections.append(rad_coef.to_markdown(index=False, floatfmt=".3f"))
            sections.append("")
    else:
        sections.append("## Experiment 3 — Chinese radical sensitivity\n\n_No cached data found._\n")

    # ---- Experiment 4 ---------------------------------------------------
    perturb = _maybe_read(root / "paraphrase_token_perturb" / "summary.csv")
    if perturb is not None and not perturb.empty:
        sections.append("## Experiment 4 — Paraphrase vs token perturbation\n")
        sections.append(perturb.to_markdown(index=False, floatfmt=".3f"))
        sections.append("")
    else:
        sections.append("## Experiment 4 — Paraphrase vs token perturbation\n\n_No cached data found._\n")

    # ---- Experiment 5 ---------------------------------------------------
    adapt = _maybe_read(root / "tokenizer_adaptation" / "audit_before_after.csv")
    if adapt is not None and not adapt.empty:
        _plot_before_after(adapt.to_dict(orient="records"),
                           root / "tokenizer_adaptation" / "plots" / "audit_before_after")
        sections.append("## Experiment 5 — Tokenizer adaptation\n")
        sections.append(adapt.to_markdown(index=False, floatfmt=".3f"))
        sections.append("")
    else:
        sections.append("## Experiment 5 — Tokenizer adaptation\n\n_No cached data found._\n")

    sections.append("## Provenance\n")
    sections.append(json.dumps({"results_dir": str(root.resolve())}, indent=2))

    write_text("\n".join(sections), report_out)
    click.echo(f"Wrote {report_out}")


if __name__ == "__main__":
    main()
