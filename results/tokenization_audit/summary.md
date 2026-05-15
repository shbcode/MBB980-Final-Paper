# Experiment 1 — Tokenization audit

**Languages:** `en` vs `zh`. **Metric:** TP = zh_tokens / en_tokens.
TP > 1 means the tokenizer takes more tokens to encode the target
language than the source language for the same content.

## Per-tokenizer summary

| tokenizer       | dataset   |   n_pairs |   mean_TP |   median_TP |   TP_ci_low |   TP_ci_high |   mean_context_shrinkage |   mean_en_tokens |   mean_zh_tokens |   wilcoxon_stat |   wilcoxon_pvalue |
|:----------------|:----------|----------:|----------:|------------:|------------:|-------------:|-------------------------:|-----------------:|-----------------:|----------------:|------------------:|
| byte            | flores200 |      1012 |     0.896 |       0.881 |       0.885 |        0.906 |                   -0.153 |          130.534 |          115.973 |       77262.000 |             0.000 |
| char            | flores200 |      1012 |     0.331 |       0.311 |       0.326 |        0.337 |                   -2.203 |          130.405 |           42.763 |           0.000 |             0.000 |
| llama3          | flores200 |      1012 |     1.313 |       1.276 |       1.297 |        1.329 |                    0.209 |           26.853 |           35.019 |       10150.000 |             0.000 |
| qwen25          | flores200 |      1012 |     1.005 |       0.966 |       0.991 |        1.020 |                   -0.042 |           27.297 |           27.437 |      194073.500 |             0.041 |
| tiktoken_cl100k | flores200 |      1012 |     1.886 |       1.851 |       1.863 |        1.910 |                    0.449 |           26.864 |           50.285 |           1.500 |             0.000 |
| tiktoken_o200k  | flores200 |      1012 |     1.281 |       1.237 |       1.265 |        1.297 |                    0.189 |           26.558 |           33.836 |       15490.500 |             0.000 |

## Reading guide

- TP near 1.0 = symmetric efficiency between the two languages.
- High TP for `zh` indicates the tokenizer fragments zh more, which
  proportionally shrinks the effective context window for `zh` users.
- *Context shrinkage* = 1 - 1/TP estimates the fraction of the
  effective context lost when switching to `zh` at fixed token budgets.

## Interpretation guardrails

- These numbers are **tokenizer-specific**. They do **not**
  themselves measure model quality or claim that `en` is
  intrinsically a 'better' language.
- Token compression, context-window consequences, and downstream task
  performance must be reported as separate axes (see Experiment 2).