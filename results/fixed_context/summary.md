# Experiment 2 — Fixed-context fairness

Backend: `dummy` / measure tokenizer: `tiktoken_cl100k`

## Per-(language, budget) summary

| language   |   budget | task   |   n_queries |   examples_fit_mean |   accuracy |    f1 |    em |   rougeL |   tokens_in |   tokens_out |   latency_total_s |   cost_total_usd |   cost_per_correct |
|:-----------|---------:|:-------|------------:|--------------------:|-----------:|------:|------:|---------:|------------:|-------------:|------------------:|-----------------:|-------------------:|
| en         |      512 | xnli   |          50 |              11.800 |      0.000 | 0.000 | 0.000 |    0.000 |       16802 |           50 |             0.001 |            0.000 |                nan |
| en         |     1024 | xnli   |          50 |              22.000 |      0.000 | 0.000 | 0.000 |    0.000 |       34472 |           50 |             0.002 |            0.000 |                nan |
| en         |     2048 | xnli   |          50 |              32.000 |      0.000 | 0.000 | 0.000 |    0.000 |       53072 |           50 |             0.004 |            0.000 |                nan |
| zh         |      512 | xnli   |          50 |               6.440 |      0.000 | 0.000 | 0.000 |    0.000 |        1273 |           50 |             0.001 |            0.000 |                nan |
| zh         |     1024 | xnli   |          50 |              17.720 |      0.000 | 0.000 | 0.000 |    0.000 |        2965 |           50 |             0.001 |            0.000 |                nan |
| zh         |     2048 | xnli   |          50 |              29.500 |      0.000 | 0.000 | 0.000 |    0.000 |        4732 |           50 |             0.002 |            0.000 |                nan |

## AUC of accuracy vs budget (trapezoidal, range-normalized)

| language   | task   |   auc_acc_vs_budget |
|:-----------|:-------|--------------------:|
| en         | xnli   |               0.000 |
| zh         | xnli   |               0.000 |

## Reading guide

- `examples_fit_mean` shows how many demonstrations the budget can hold per language
  with the *measure tokenizer*. Differences here are a direct expression of
  context-window economics, independent of model quality.
- `accuracy` (or `f1`, `em`, `rougeL`) at each budget reflects the *combined* effect
  of demonstration availability and any underlying model behaviour.
- `cost_per_correct` translates the same axis into dollar terms when a hosted backend
  is used.

## Guardrails

- Differences are **task- and tokenizer-specific**. Do not generalise to broad claims
  about language ability.
- For interpretation, pair these results with the Experiment 1 audit so the reader
  can separate *compression* from *downstream* effects.