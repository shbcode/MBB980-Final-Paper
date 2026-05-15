# `reports/paper_assets/` — paper-ready figures and tables

Everything in this directory is regenerated deterministically by

```
python -m src.build_paper_assets
```

from the cached experiment CSVs. The build script never re-runs an
experiment and never makes a network call.

## Inputs

| Asset | Cached input(s) |
|---|---|
| Figure 1 / Table 1 / Appendix Table J | `results/tokenization_audit/summary_by_tokenizer.csv` |
| Figure 2 / Appendix Figure G / Appendix Table F | `results/fixed_context/summary.csv` |
| Table 2 / Appendix coefficient table | `results/radical_sensitivity/contingency_same_radical.csv`, `results/radical_sensitivity/logit_coefficients.csv` |
| Figure 3 | `results/tokenizer_adaptation/audit_before_after.csv` |
| Appendix Table H / Appendix Figure I | `results/paraphrase_token_perturb/summary.csv`, optional row-level `perturbations.csv` |

If any required input is missing, the build fails loudly with the expected
path printed in the error message. Re-run the corresponding experiment
(see the top-level `README.md`) before retrying.

## Outputs

Every figure is written as both `.png` (220 dpi) and `.pdf`. Every table is
written as `.csv` (raw numerical), `.md` (publication-readable), and `.tex`
(simple `tabular` for paste-into-paper).

See `captions.md` for first-draft caption text for each asset.

## Assumptions and fallbacks

- Tokenizer ordering follows the worst-parity-rightward convention:
  `char, byte, qwen25, tiktoken_o200k, llama3, tiktoken_cl100k`. Missing
  tokenizers are dropped silently; unknown ones are appended at the end.
- The perturbation appendix figure prefers `perturbations.csv` (row-level,
  shown as box plots). If that file is absent, it falls back to a bar chart
  of summary means.
- Coefficients flagged as `regularized` in `logit_coefficients.csv` are
  rendered without p-values (none are estimable from a regularised fit) and
  Table 2's caption is amended to note the fallback.
