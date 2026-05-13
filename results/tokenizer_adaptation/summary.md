# Experiment 5 — Tokenizer adaptation (lightweight)

Base tokenizer: `qwen25`

## Counterfactual audit (added vocabulary, no retraining)

|   added_tokens |   mean_en_tokens |   mean_zh_tokens |   mean_TP |
|---------------:|-----------------:|-----------------:|----------:|
|          0.000 |           28.265 |           29.760 |     1.049 |
|        326.000 |           28.265 |           31.710 |     1.118 |
|        326.000 |           28.265 |           31.710 |     1.118 |
|        326.000 |           28.265 |           31.710 |     1.118 |

## Reading guide

- The `0` row is the unmodified base tokenizer's audit.
- Each subsequent row simulates *if* the new vocabulary existed: it
  reports the lower-bound effect on zh token counts (greedy longest match).
- A real trained extension typically lands between this counterfactual and
  the base, depending on training quality.

## Continued pretraining

Skipped (`do_train: false`).
