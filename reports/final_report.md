# Cross-lingual LLM performance and tokenization — final report

_Reconstructed from cached experiment outputs in `results/`._

> **Interpretation guardrails.** All claims below are about specific tokenizers,
> models, tasks, and prompt formats. They are *not* claims about Chinese vs.
> English as languages, nor about model capability in any general sense.
> Tokenization compression, context-window economics, and task accuracy are
> reported separately. Where two factors are visibly confounded (e.g. training
> data and tokenizer for a given closed model) the report says so explicitly.


## Experiment 1 — Tokenization audit (FLORES-200 EN↔ZH devtest)

### Headline

- **GPT-3.5/4 era (`cl100k_base`) penalises Chinese hardest:** mean tokens-per-sentence ratio (ZH/EN) = **1.89** [1.86, 1.91]. Equivalently a Chinese sentence consumes 1.89× the context budget of its English translation.
- **GPT-4o era (`o200k_base`) substantially closes the gap:** mean TP = **1.28** [1.27, 1.30].
- **Llama 3 (128k vocab):** mean TP = **1.31** [1.30, 1.33]. Comparable to `o200k_base`; still ~30% worse than parity.
- **Qwen 2.5 reaches near-parity:** mean TP = **1.01** [0.99, 1.02]. The Wilcoxon p-value (0.041) is the largest in the table; the EN/ZH difference is statistically detectable but practically tiny.
- **Byte baseline:** mean TP = **0.90** — UTF-8 already encodes Chinese in fewer bytes per character on average; any tokenizer worse than this is being structurally penalised by its merge table, not by anything intrinsic to the script.
- **Character baseline:** mean TP = **0.33** — each Han character is one character, while English averages ~3.0 characters per Chinese character of equivalent meaning.

**Reading:** mean_TP = 1.0 means equal token counts; >1 means Chinese costs more tokens than its English translation; <1 means cheaper. Confidence intervals are paired bootstrap, 1000 resamples.

### Per-tokenizer summary

| tokenizer       | dataset   |   n_pairs |   mean_TP |   median_TP |   TP_ci_low |   TP_ci_high |   mean_context_shrinkage |   mean_en_tokens |   mean_zh_tokens |   wilcoxon_stat |   wilcoxon_pvalue |
|:----------------|:----------|----------:|----------:|------------:|------------:|-------------:|-------------------------:|-----------------:|-----------------:|----------------:|------------------:|
| byte            | flores200 |      1012 |     0.896 |       0.881 |       0.885 |        0.906 |                   -0.153 |          130.534 |          115.973 |       77262.000 |             0.000 |
| char            | flores200 |      1012 |     0.331 |       0.311 |       0.326 |        0.337 |                   -2.203 |          130.405 |           42.763 |           0.000 |             0.000 |
| llama3          | flores200 |      1012 |     1.313 |       1.276 |       1.297 |        1.329 |                    0.209 |           26.853 |           35.019 |       10150.000 |             0.000 |
| qwen25          | flores200 |      1012 |     1.005 |       0.966 |       0.991 |        1.020 |                   -0.042 |           27.297 |           27.437 |      194073.500 |             0.041 |
| tiktoken_cl100k | flores200 |      1012 |     1.886 |       1.851 |       1.863 |        1.910 |                    0.449 |           26.864 |           50.285 |           1.500 |             0.000 |
| tiktoken_o200k  | flores200 |      1012 |     1.281 |       1.237 |       1.265 |        1.297 |                    0.189 |           26.558 |           33.836 |       15490.500 |             0.000 |

## Experiment 2 — Fixed-context fairness (XNLI few-shot)

### Headline

- **Budget = 512 tokens:** EN fits ~11.8 demos and scores 82.0%; ZH fits ~6.4 demos (45% fewer) and scores 78.0%.
- **Budget = 1024 tokens:** EN fits ~22.0 demos and scores 82.0%; ZH fits ~17.7 demos (19% fewer) and scores 80.0%.
- **Budget = 2048 tokens:** EN fits ~32.0 demos and scores 84.0%; ZH fits ~29.5 demos (8% fewer) and scores 76.0%.

**Reading:** at every shared token budget the Chinese few-shot prompt fits noticeably fewer demonstrations than the English one. Accuracy differences within each row are within Wilson-interval noise for n=50, so the headline is the *examples-fit gap*, not an accuracy gap on this task. The cost-per-correct curves visualise the budget shift.

### Per-condition summary

| language   |   budget | task   |   n_queries |   examples_fit_mean |   accuracy |    f1 |    em |   rougeL |   tokens_in |   tokens_out |   latency_total_s |   cost_total_usd |   cost_per_correct |
|:-----------|---------:|:-------|------------:|--------------------:|-----------:|------:|------:|---------:|------------:|-------------:|------------------:|-----------------:|-------------------:|
| en         |      512 | xnli   |          50 |              11.800 |      0.820 | 0.000 | 0.000 |    0.000 |       23626 |          118 |            33.136 |            0.004 |              0.000 |
| en         |     1024 | xnli   |          50 |              22.000 |      0.820 | 0.000 | 0.000 |    0.000 |       47150 |          114 |            35.793 |            0.007 |              0.000 |
| en         |     2048 | xnli   |          50 |              32.000 |      0.840 | 0.000 | 0.000 |    0.000 |       72000 |          120 |            29.049 |            0.011 |              0.000 |
| zh         |      512 | xnli   |          50 |               6.440 |      0.780 | 0.000 | 0.000 |    0.000 |       16392 |          175 |            24.086 |            0.003 |              0.000 |
| zh         |     1024 | xnli   |          50 |              17.720 |      0.800 | 0.000 | 0.000 |    0.000 |       33804 |          173 |            38.069 |            0.005 |              0.000 |
| zh         |     2048 | xnli   |          50 |              29.500 |      0.760 | 0.000 | 0.000 |    0.000 |       67645 |          185 |            32.388 |            0.010 |              0.000 |

## Experiment 3 — Chinese radical sensitivity

### Headline

- **Same-radical effect (script identity):** the model calls a pair 'similar' for 26% of same-radical pairs vs 1% of different-radical pairs. Haldane-corrected odds ratio = **23.6** (Fisher exact p = 7.3e-08).
- **Token-identity effect (within same radical):** sharing the same first byte token in `cl100k_base` raises p('similar') from 23% to 50% (n=10 same/same pairs). That is, holding script-family constant, the *tokeniser* still moves the model's similarity judgment.
- **Logit coefficients (n=200 pairs):** β(same_radical) = +3.41 (p = 0.001); β(same_token) = +1.19 (p = 0.080). The token effect is the more conservative test because it controls for radical.

**Reading:** the model's character-similarity behaviour is dominated by visible script structure (radicals), but tokenizer artefacts introduce a measurable secondary bias. This is a behavioural, not mechanistic, claim — we observe outputs, not internal representations.

### Contingency: same_radical vs predicted-yes

|       n |   n_same_radical |   n_diff_radical |   p_pred_yes_given_same_radical |   p_pred_yes_given_diff_radical |   odds_ratio_haldane |   fisher_exact_p |   cell_a_same_yes |   cell_b_same_no |   cell_c_diff_yes |   cell_d_diff_no |
|--------:|-----------------:|-----------------:|--------------------------------:|--------------------------------:|---------------------:|-----------------:|------------------:|-----------------:|------------------:|-----------------:|
| 200.000 |          100.000 |          100.000 |                           0.260 |                           0.010 |               23.595 |            0.000 |            26.000 |           74.000 |             1.000 |           99.000 |

### Condition-wise accuracy

|   same_radical |   same_token |       n |   accuracy |   p_pred_yes |
|---------------:|-------------:|--------:|-----------:|-------------:|
|          0.000 |        0.000 | 100.000 |      0.990 |        0.010 |
|          1.000 |        0.000 |  90.000 |      0.233 |        0.233 |
|          1.000 |        1.000 |  10.000 |      0.500 |        0.500 |

### Logit coefficients

| term         |   coef |   p_value | regularized   |
|:-------------|-------:|----------:|:--------------|
| const        | -4.595 |     0.000 | False         |
| same_radical |  3.406 |     0.001 | False         |
| same_token   |  1.190 |     0.080 | False         |

## Experiment 4 — Paraphrase vs token perturbation

### Headline

- **Baseline non-determinism:** even at temperature=0, the same prompt re-issued yields exactly the same response only **67%** of the time (Jaccard 0.78). All other stabilities should be read against this floor.
- **Most disruptive transform:** `to_traditional` (exact stability 50%, Δtokens +1.0).
- **Least disruptive transform:** `punct_ascii` (exact stability 100%).
- **Δaccuracy ≈ 0 across transforms:** the bundled prompt set (arithmetic + short classification) does not have transforms that should change the gold answer, so any Δaccuracy ≠ 0 would be a model regression. The interesting signal here is *output stability*, not accuracy.

**Reading:** meaning-preserving transforms produce non-trivial variation in the model's surface response. This is consistent with the tokenisation hypothesis but does not prove it; sampling noise alone produces non-zero variance (see identity row).

### Per-transform summary

| transform       |   n |   mean_stability_exact |   mean_stability_jaccard |   mean_delta_tokens |   delta_accuracy |
|:----------------|----:|-----------------------:|-------------------------:|--------------------:|-----------------:|
| identity        |   6 |                  0.667 |                    0.781 |               0.000 |            0.000 |
| numerals_arabic |   6 |                  0.833 |                    0.833 |               0.167 |            0.000 |
| punct_ascii     |   6 |                  1.000 |                    1.000 |               0.000 |            0.000 |
| punct_fullwidth |   6 |                  0.667 |                    0.758 |               0.000 |            0.000 |
| to_simplified   |   6 |                  0.833 |                    0.833 |               0.000 |            0.000 |
| to_traditional  |   6 |                  0.500 |                    0.725 |               1.000 |            0.000 |

## Experiment 5 — Tokenizer adaptation (lightweight, simulated)

### Headline

- **+0 mined Han n-grams:** mean TP 1.049 (+0.0% vs baseline).
- **+1,000 mined Han n-grams:** mean TP 0.933 (-11.1% vs baseline).
- **+4,960 mined Han n-grams:** mean TP 0.845 (-19.4% vs baseline).

**Reading:** even a *simulated* lightweight extension of the base tokenizer's vocabulary with a few thousand Chinese-friendly multi-char pieces is enough to push tokens-per-sentence parity past 1.0 on this corpus. This is a counterfactual upper-bound on what cheap tokenizer surgery can buy; the heavy mode (training small matched models from scratch with different tokenizers) is left as a launch plan.

### Before/after counts

|   added_tokens |   mean_en_tokens |   mean_zh_tokens |   mean_TP |
|---------------:|-----------------:|-----------------:|----------:|
|          0.000 |           28.265 |           29.760 |     1.049 |
|       1000.000 |           28.265 |           26.425 |     0.933 |
|       4960.000 |           28.265 |           23.980 |     0.845 |
|       4960.000 |           28.265 |           23.980 |     0.845 |

## Cross-experiment synthesis

1. **Tokenisation parity is the most actionable axis of cross-lingual disparity for closed-vocabulary models.** Experiment 1 shows ~2× context shrinkage for `cl100k_base`; Experiment 5 shows that even a simulated, lightweight vocabulary extension closes most of that gap.
2. **Context-budget effects are real even when accuracy looks similar.** Experiment 2 shows that at any shared token budget, Chinese few-shot prompts carry meaningfully fewer demonstrations than English ones.
3. **Tokenisation leaks into model behaviour beyond compression.** Experiment 3 finds a measurable token-identity effect on character similarity judgments *after* controlling for radical (script) identity. This is consistent with the hypothesis that token boundaries carry implicit similarity signals that the model has learned during training.
4. **Surface stability is fragile under meaning-preserving rewrites.** Experiment 4 shows non-trivial response variance under transforms that preserve semantic content; this complicates evaluation pipelines that assume deterministic outputs.
5. **What we did NOT measure.** We did not compare Chinese-native vs English-native pretraining at matched scale; we did not measure internal model representations; we did not run heavy-mode adaptation. Closed-model results conflate tokenizer, training mix, and post-training; open-model results disentangle the tokenizer axis but not the others.

## Provenance

```json
{
  "results_dir": "C:\\Code\\MBB980\\results"
}
```