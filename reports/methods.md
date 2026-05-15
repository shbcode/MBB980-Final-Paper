# Methods

This document describes the experimental design, data pipeline, and analysis
choices used in the cross-linguistic LLM tokenization study. It is written so
that an external reader can reproduce the entire pipeline from this repository
plus the listed datasets and model checkpoints.

## 1. Datasets

| Dataset    | Splits used               | Languages          | Purpose                          |
|------------|---------------------------|--------------------|----------------------------------|
| FLORES-200 | `dev`, `devtest`          | EN, ZH (+optional) | Audit + adaptation eval          |
| XNLI       | `validation`              | EN, ZH             | Few-shot classification          |
| XQuAD      | `validation`              | EN, ZH             | Few-shot extractive QA           |
| MGSM       | `test`                    | EN, ZH             | Optional reasoning task          |
| XLSUM      | `validation`              | EN, ZH             | Optional summarization           |

Each row is normalized to the schema `{id, dataset, split, domain, language,
text, paired_id}`. Bilingual alignment is asserted by `paired_id`; any
misalignment fails loudly during loading.

## 2. Tokenizers

We compare:

- **tiktoken cl100k_base** (proxy for GPT-4 family) and `o200k_base` if available.
- **Llama 3** tokenizer (`meta-llama/Meta-Llama-3.1-8B-Instruct`).
- **Qwen 2.5** tokenizer (`Qwen/Qwen2.5-7B-Instruct`).
- **SentencePiece BPE** and **Unigram** trained on the bilingual corpus
  (`src/tokenizers/train_sentencepiece.py`).
- **Byte baseline** (UTF-8 bytes) — lower bound on length.
- **Character baseline** (Unicode codepoints).

All tokenizers expose a uniform `encode(text) -> list[int]` interface
(`src/tokenizers/registry.py`).

## 3. Normalization

Implemented in `src/data/normalize.py`. We expose orthogonal switches:
NFKC unicode, full-width <-> ASCII punctuation, numeric format, and
OpenCC-driven Simplified <-> Traditional Chinese. The default for the audit is
`unicode_nfkc=true, punctuation=false, numbers=false, simplify=false,
collapse_whitespace=true` so that any disparity is *not* the by-product of
asymmetric cleanup.

## 4. Statistics

- Bootstrap (percentile) confidence intervals with 2 000 resamples by default;
  paired bootstrap when comparing aligned series (`src/stats/bootstrap.py`).
- Wilcoxon signed-rank for paired EN/ZH token-count comparisons within each
  tokenizer (`src/stats/tests.py`).
- Logistic regression (`statsmodels`) on the radical-sensitivity task to
  recover coefficients for `same_radical` and `same_token`.

## 5. Experiments

### 5.1 Tokenization audit (Experiment 1)

For every aligned pair and every tokenizer we compute tokens, chars-per-token,
bytes-per-token, **TP** = `zh_tokens / en_tokens`, and **context shrinkage**
= `1 - 1/TP`. Outputs include per-sentence and per-tokenizer summaries with
bootstrap CIs, plus bar/box/scatter plots.

### 5.2 Fixed-context fairness (Experiment 2)

For each language and budget B, we greedily pack as many demonstrations into a
prompt as fit (measured by the *served* model's tokenizer), then query the
backend, score with the task metric, and aggregate. We report `examples_fit`,
`accuracy` (or `f1`/`em`/`rougeL`), and `cost_per_correct`. Optional ablations:
`equal_demo_count` (same K demonstrations regardless of language) and
`byte_normalized` (budget rebalanced by bytes-per-token).

### 5.3 Chinese radical sensitivity (Experiment 3)

Triples and pairs are constructed with controlled crossings of `same_radical`
and `same_token` from a 108-character bilingual seed CSV
(`data/processed/radical_chars.csv`) covering ten common radicals at HSK 1-4
frequency. We use `tiktoken_cl100k_base` as the `measure_tokenizer` because
it fragments many Han characters into 2-3 byte tokens, providing real
variance in `same_token`; tokenizers that map every Han character to its own
id (Qwen, Llama 3) leave `same_token` constant and statistically uninformative.

We run similarity judgment and odd-one-out at temperature 0 and fit logistic
regressions to separate radical-driven from tokenizer-driven judgments. As a
robustness check we also report a Haldane-corrected odds ratio plus Fisher's
exact p-value on the `(same_radical, pred_yes)` 2x2 table, which is stable
under perfect/near-perfect separation. When the MLE fit fails to converge we
fall back to L2-regularised logistic regression and flag the row.

### 5.4 Paraphrase vs token perturbation (Experiment 4)

For a project-supplied prompt set, we apply meaning-preserving paraphrases and
surface-level token perturbations (punctuation, numerals, simplified/traditional,
synonyms). We compare token counts, response stability (exact match + Jaccard
on response tokens), and accuracy delta.

### 5.5 Tokenizer adaptation (Experiment 5)

**Lightweight mode** mines Han n-grams of length 1-4 from the adaptation
corpus that the base tokenizer fragments (`min_count=2` to keep the candidate
pool wide enough to differentiate 1k/5k/10k budgets on small corpora). The
counterfactual token count after extension is computed by subtracting
per-occurrence savings — each occurrence of a piece that the base tokenizer
splits into `K` tokens saves `K-1` tokens once the piece becomes one token —
using longest-match-first replacement. This is a fast proxy for true
embedding extension; the assumption that pieces tokenise as exactly one
token after addition is the standard BPE-merge assumption.

It optionally also extends an HF tokenizer + resizes the embedding table +
runs a short continued-pretraining loop on a small EN/ZH mix.

**Heavy mode** prints a launch plan for training small matched decoder-only
models from scratch with different tokenizers; actual training is delegated to
an external trainer (e.g. nanoGPT, litgpt).

## 6. Reproducibility

Each CLI invocation creates a timestamped run directory under
`<output_dir>/runs/<ts>__<name>/`, persisting the effective config, the git
commit, the `manifest.json` (seed, env, elapsed), and a `run.log`.
`src/reconstruct_analysis.py` rebuilds plots and `reports/final_report.md`
purely from cached CSVs.

## 7. Threats to validity

- **Tokenizer-specific results.** The audit is, by definition, conditional on
  the tokenizer family; our SentencePiece baselines depend on the corpus they
  were trained on.
- **Task-specific results.** Differences in fixed-context fairness reflect
  task structure and prompt template choices, not pure language ability.
- **Model-specific results.** All radical-sensitivity and perturbation
  effects are measured for one model at a time; cross-model claims require
  re-running with each backend.
- **Training-data confounds.** Pre-training data mixture is unknown for
  closed models; a smaller ZH improvement after adaptation may reflect
  pre-existing ZH coverage rather than tokenization gains.

We do not claim that one language is intrinsically easier or harder; we only
quantify how *tokenization*, *context economics*, *training/interface*, and
*downstream task* axes interact.
