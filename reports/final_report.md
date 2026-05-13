# Final report (reconstructed from cache)

## Experiment 1 — Tokenization audit

| tokenizer       | dataset   |   n_pairs |   mean_TP |   median_TP |   TP_ci_low |   TP_ci_high |   mean_context_shrinkage |   mean_en_tokens |   mean_zh_tokens |   wilcoxon_stat |   wilcoxon_pvalue |
|:----------------|:----------|----------:|----------:|------------:|------------:|-------------:|-------------------------:|-----------------:|-----------------:|----------------:|------------------:|
| byte            | flores200 |      1012 |     0.896 |       0.881 |       0.885 |        0.906 |                   -0.153 |          130.534 |          115.973 |       77262.000 |             0.000 |
| char            | flores200 |      1012 |     0.331 |       0.311 |       0.326 |        0.337 |                   -2.203 |          130.405 |           42.763 |           0.000 |             0.000 |
| qwen25          | flores200 |      1012 |     1.005 |       0.966 |       0.991 |        1.020 |                   -0.042 |           27.297 |           27.437 |      194073.500 |             0.041 |
| tiktoken_cl100k | flores200 |      1012 |     1.886 |       1.851 |       1.863 |        1.910 |                    0.449 |           26.864 |           50.285 |           1.500 |             0.000 |
| tiktoken_o200k  | flores200 |      1012 |     1.281 |       1.237 |       1.265 |        1.297 |                    0.189 |           26.558 |           33.836 |       15490.500 |             0.000 |

## Experiment 2 — Fixed-context fairness

| language   |   budget | task   |   n_queries |   examples_fit_mean |   accuracy |    f1 |    em |   rougeL |   tokens_in |   tokens_out |   latency_total_s |   cost_total_usd |   cost_per_correct |
|:-----------|---------:|:-------|------------:|--------------------:|-----------:|------:|------:|---------:|------------:|-------------:|------------------:|-----------------:|-------------------:|
| en         |      512 | xnli   |          50 |              11.800 |      0.000 | 0.000 | 0.000 |    0.000 |       16802 |           50 |             0.001 |            0.000 |                nan |
| en         |     1024 | xnli   |          50 |              22.000 |      0.000 | 0.000 | 0.000 |    0.000 |       34472 |           50 |             0.002 |            0.000 |                nan |
| en         |     2048 | xnli   |          50 |              32.000 |      0.000 | 0.000 | 0.000 |    0.000 |       53072 |           50 |             0.004 |            0.000 |                nan |
| zh         |      512 | xnli   |          50 |               6.440 |      0.000 | 0.000 | 0.000 |    0.000 |        1273 |           50 |             0.001 |            0.000 |                nan |
| zh         |     1024 | xnli   |          50 |              17.720 |      0.000 | 0.000 | 0.000 |    0.000 |        2965 |           50 |             0.001 |            0.000 |                nan |
| zh         |     2048 | xnli   |          50 |              29.500 |      0.000 | 0.000 | 0.000 |    0.000 |        4732 |           50 |             0.002 |            0.000 |                nan |

## Experiment 3 — Chinese radical sensitivity

### Condition-wise accuracy

|   same_radical |   same_token |      n |   accuracy |   p_pred_yes |
|---------------:|-------------:|-------:|-----------:|-------------:|
|          0.000 |        0.000 | 40.000 |      1.000 |        0.000 |
|          1.000 |        0.000 | 15.000 |      0.000 |        0.000 |

## Experiment 4 — Paraphrase vs token perturbation

| transform       |   n |   mean_stability_exact |   mean_stability_jaccard |   mean_delta_tokens |   delta_accuracy |
|:----------------|----:|-----------------------:|-------------------------:|--------------------:|-----------------:|
| identity        |   6 |                  1.000 |                    1.000 |               0.000 |            0.000 |
| numerals_arabic |   6 |                  1.000 |                    1.000 |               0.167 |            0.000 |
| punct_ascii     |   6 |                  1.000 |                    1.000 |               0.000 |            0.000 |
| punct_fullwidth |   6 |                  1.000 |                    1.000 |               0.000 |            0.000 |
| to_simplified   |   6 |                  1.000 |                    1.000 |               0.000 |            0.000 |
| to_traditional  |   6 |                  1.000 |                    1.000 |               1.000 |            0.000 |

## Experiment 5 — Tokenizer adaptation

|   added_tokens |   mean_en_tokens |   mean_zh_tokens |   mean_TP |
|---------------:|-----------------:|-----------------:|----------:|
|          0.000 |           28.265 |           29.760 |     1.049 |
|        326.000 |           28.265 |           31.710 |     1.118 |
|        326.000 |           28.265 |           31.710 |     1.118 |
|        326.000 |           28.265 |           31.710 |     1.118 |

## Provenance

{
  "results_dir": "C:\\Code\\MBB980\\results"
}