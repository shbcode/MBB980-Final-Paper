# Experiment 3 — Chinese radical sensitivity

## Condition-wise accuracy (similarity task)

|   same_radical |   same_token |      n |   accuracy |   p_pred_yes |
|---------------:|-------------:|-------:|-----------:|-------------:|
|          0.000 |        0.000 | 40.000 |      1.000 |        0.000 |
|          1.000 |        0.000 | 15.000 |      0.000 |        0.000 |

## Logistic regression coefficients

_Regression skipped._

## Error analysis notes

- Inspect `raw.jsonl` for cases where `pred_yes=1` but `same_radical=0`
  (model overgeneralises) or `pred_yes=0` and `same_radical=1` (model misses).
- A non-zero `same_token` coefficient is evidence that *tokenizer* identity,
  not just *script* identity, modulates the model's similarity judgment.