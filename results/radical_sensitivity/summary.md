# Experiment 3 — Chinese radical sensitivity

## Condition-wise accuracy (similarity task)

|   same_radical |   same_token |       n |   accuracy |   p_pred_yes |
|---------------:|-------------:|--------:|-----------:|-------------:|
|          0.000 |        0.000 | 100.000 |      0.990 |        0.010 |
|          1.000 |        0.000 |  90.000 |      0.233 |        0.233 |
|          1.000 |        1.000 |  10.000 |      0.500 |        0.500 |

## Contingency: same_radical vs pred_yes

|       n |   n_same_radical |   n_diff_radical |   p_pred_yes_given_same_radical |   p_pred_yes_given_diff_radical |   odds_ratio_haldane |   fisher_exact_p |   cell_a_same_yes |   cell_b_same_no |   cell_c_diff_yes |   cell_d_diff_no |
|--------:|-----------------:|-----------------:|--------------------------------:|--------------------------------:|---------------------:|-----------------:|------------------:|-----------------:|------------------:|-----------------:|
| 200.000 |          100.000 |          100.000 |                           0.260 |                           0.010 |               23.595 |            0.000 |            26.000 |           74.000 |             1.000 |           99.000 |

Odds ratio uses a Haldane-Anscombe (+0.5) correction; Fisher's exact p-value is unconditional and exact. These are stable on small samples where logistic regression suffers from perfect/near-perfect separation.

## Logistic regression coefficients

| term         |   coef |   p_value | regularized   |
|:-------------|-------:|----------:|:--------------|
| const        | -4.595 |     0.000 | False         |
| same_radical |  3.406 |     0.001 | False         |
| same_token   |  1.190 |     0.080 | False         |

## Error analysis notes

- Inspect `raw.jsonl` for cases where `pred_yes=1` but `same_radical=0`
  (model overgeneralises) or `pred_yes=0` and `same_radical=1` (model misses).
- A non-zero `same_token` coefficient is evidence that *tokenizer* identity,
  not just *script* identity, modulates the model's similarity judgment.
- Tiny seed datasets (~13 chars) leave `same_token` constant for HF
  tokenizers that map every Han char to its own id. Supply a richer
  `dataset_csv` (e.g. ~200 chars across ~10 radicals) to recover power.