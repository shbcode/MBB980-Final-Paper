# Caption drafts for `reports/paper_assets/`

These are sober, paste-ready first-draft captions for each figure and table.
Adjust to match your manuscript's prose style; the numerical claims should be
left intact unless the underlying CSVs change.

## Main text

### Figure 1 — `fig1_tokenization_parity_by_tokenizer.{png,pdf}`
Mean tokenization parity (ZH / EN) on FLORES-200 EN-ZH devtest (N = 1012),
broken out by tokenizer. The dashed line marks parity (TP = 1.0); error bars
show 95% paired-bootstrap confidence intervals. Chinese incurs a much larger
token burden under `cl100k_base` than under `o200k_base`, Llama 3, or
Qwen 2.5 — a tokenizer-dependent gap, not a property of the language itself.
Hatched bars highlight tokenizers that deviate noticeably above (`///`) or
below (`...`) parity.

### Table 1 — `table1_tokenization_audit_summary.{csv,md,tex}`
Per-tokenizer summary on FLORES-200 EN-ZH devtest: mean tokenization parity
(ZH / EN) with 95% paired-bootstrap CI, and mean token counts per language.
Statistical extras (Wilcoxon, context shrinkage) are deferred to the
appendix table.

### Figure 2 — `fig2_examples_fit_vs_budget.{png,pdf}`
Mean number of few-shot demonstrations that fit inside a fixed token budget
on XNLI, by language. At every shared budget the Chinese prompt fits fewer
demonstrations than the English one. The figure visualises the *context-budget
consequence* of the parity gap in Figure 1; it is not an accuracy claim.

### Table 2 — `table2_radical_token_sensitivity_summary.{csv,md,tex}`
Behavioral evidence that token identity modulates the model's
character-similarity judgments. We report the conditional probability that
the model calls a pair "similar" given same vs. different radical, the
Haldane-Anscombe-corrected odds ratio with Fisher's exact p-value, and
logit coefficients from a regression that controls for radical when
estimating the token-identity effect. Coefficients are behavioral
estimates: they describe input/output regularities, not internal model
representations. Token-identity is computed against `cl100k_base` because
single Han characters fragment into byte tokens under that vocabulary.

### Figure 3 — `fig3_tokenizer_adaptation_counterfactual.{png,pdf}`
Simulated counterfactual audit: mean tokenization parity (ZH / EN) on the
same FLORES-200 corpus after extending the base tokenizer's vocabulary with
0, 1k, and ~5k mined Han n-grams. The dashed line marks parity. This is a
*lightweight, simulated* intervention — we recompute token counts under the
extended vocabulary; we do not retrain or fine-tune the model. The purpose
is to bound how much of the disparity is interface-modifiable.

## Appendix

### Appendix Figure G — `appendix_fig_accuracy_vs_budget.{png,pdf}`
XNLI accuracy at fixed token budgets, by language. Within-row differences
are within Wilson-interval noise for n = 50; we report this for completeness.

### Appendix Table F — `appendix_table_fixed_context_full.{csv,md,tex}`
Full Experiment 2 summary including demonstrations fit, accuracy, cost per
correct prediction, query count, and total in/out tokens for every
(language, budget) cell.

### Appendix Table H — `appendix_table_perturbation_summary.{csv,md,tex}`
Per-transform summary of Experiment 4. The `identity` row is the
temperature = 0 non-determinism floor and should be read as a baseline
against which other rows are interpreted. Δaccuracy ≈ 0 is expected on this
prompt set (the gold answer is preserved by every transform); the
informative quantity is *output stability*.

### Appendix Figure I — `appendix_fig_perturbation_stability.{png,pdf}`
Per-prompt response stability (Jaccard) by transform. Box plots when the
row-level perturbation log is available; bar of means otherwise.

### Appendix Table J — `appendix_table_full_audit_with_controls.{csv,md,tex}`
Full per-tokenizer audit including context shrinkage (1 - 1/TP) and Wilcoxon
signed-rank p-values testing EN-vs-ZH token-count equality within each
tokenizer.

### Appendix Table — `appendix_table_radical_logit_coefficients.{csv,md,tex}`
Raw logistic-regression coefficients for Experiment 3.

## Cautionary notes for all assets

- These are claims about specific tokenizers, models, tasks, and prompt
  formats. They are not claims about the relative difficulty of any human
  language or about model capability in general.
- Closed-vocabulary closed-model results conflate tokenizer, training mix,
  and post-training; the open-model audit (Llama 3, Qwen 2.5) isolates the
  tokenizer axis but not the others.
- Experiment 5's "added tokens" are a counterfactual measurement (we
  recompute token counts), not a fully retrained adapted model.
