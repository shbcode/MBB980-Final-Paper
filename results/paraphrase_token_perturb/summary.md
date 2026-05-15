# Experiment 4 — Paraphrase vs token perturbation

## Per-transform summary

| transform       |   n |   mean_stability_exact |   mean_stability_jaccard |   mean_delta_tokens |   delta_accuracy |
|:----------------|----:|-----------------------:|-------------------------:|--------------------:|-----------------:|
| identity        |   6 |                  0.667 |                    0.781 |               0.000 |            0.000 |
| numerals_arabic |   6 |                  0.833 |                    0.833 |               0.167 |            0.000 |
| punct_ascii     |   6 |                  1.000 |                    1.000 |               0.000 |            0.000 |
| punct_fullwidth |   6 |                  0.667 |                    0.758 |               0.000 |            0.000 |
| to_simplified   |   6 |                  0.833 |                    0.833 |               0.000 |            0.000 |
| to_traditional  |   6 |                  0.500 |                    0.725 |               1.000 |            0.000 |

## Reading guide

- `mean_stability_jaccard` near 1.0 means the model produces a near-identical
  response despite the transform. Lower values indicate sensitivity.
- `delta_accuracy` is positive when the transform helps (e.g. cleaner
  numerals) and negative when it hurts.
- Compare paraphrase rows (semantic-preserving) against token-perturbation
  rows: a model that's more sensitive to surface tokenization than to
  meaning is leaning on tokenization artefacts.
