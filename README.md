# Cross-linguistic LLM Performance and Tokenization

A reproducible research pipeline that quantifies how much of the
English-vs-Chinese performance disparity in large language models is
attributable to **tokenization**, **context-window economics**, and
**training-data / interface** choices, with optional extension to Japanese,
Arabic, Hindi, and Tamil.

The pipeline produces clean artifacts: CSVs, plots (PNG + PDF), markdown
summaries, run logs, and a final analysis notebook.

---

## TL;DR

```bash
# 0. Set up the environment (Python 3.11)
uv venv && source .venv/bin/activate     # or: python -m venv .venv && source .venv/bin/activate
uv pip install -e .                      # core deps; runs Experiments 1, 4 (offline) and the audit
uv pip install -e '.[hf]'                # add local HF model support (Experiments 2, 3, 5)
uv pip install -e '.[api]'               # add OpenAI / Together hosted backend
uv pip install -e '.[notebook]'          # for the final notebook

# 1. Run the five experiments (each is independent; each saves CSVs + plots)
python -m src.run_tokenization_audit  --config configs/audit.yaml
python -m src.run_fixed_context        --config configs/fixed_context.yaml
python -m src.run_radical_sensitivity  --config configs/radicals.yaml
python -m src.run_paraphrase_perturb   --config configs/perturb.yaml
python -m src.run_tokenizer_adaptation --config configs/adaptation.yaml

# 2. Rebuild reports / plots from cached CSVs at any time
python -m src.reconstruct_analysis --results-dir results --report-out reports/final_report.md

# 3. Build publication-ready figures and tables (PNG/PDF + CSV/MD/TeX)
python -m src.build_paper_assets --results-dir results --out reports/paper_assets

# 4. Open the notebook
jupyter lab notebooks/final_analysis.ipynb
```

`src.build_paper_assets` reads only cached CSVs (no network, no model calls)
and emits paper-ready figures and tables under `reports/paper_assets/`,
including caption drafts in `captions.md` and a per-asset README. See
`reports/paper_assets/README.md` for the input/output mapping.

If you have neither HF model access nor API keys, the experiments still run
end-to-end against a `dummy` backend; the tokenization audit (Experiment 1)
and the perturbation experiment (Experiment 4) produce real, meaningful
numbers without any model call.

---

## Project layout

```
configs/                YAML config per experiment
data/raw/               cached corpora (FLORES-200, XNLI, XQuAD, MGSM, XLSUM)
data/processed/         normalized bilingual datasets + supplied prompt CSVs
notebooks/              final_analysis.ipynb
reports/                methods.md, final_report.md
results/<experiment>/   CSVs, summary.md, plots/, runs/<timestamp>__<name>/
src/                    modular code (see below)
tests/                  pytest smoke tests
```

### Source layout

```
src/
  utils/        config, logging, IO, run-management
  data/         bilingual loaders, normalization
  tokenizers/   uniform tokenizer registry (tiktoken, HF, SentencePiece, byte, char)
  backends/     model backends (HF local, OpenAI/Together, dummy)
  stats/        bootstrap CIs, paired Wilcoxon
  plotting/     publication matplotlib defaults (no seaborn)
  experiments/  the five experiments
  run_*.py      CLI entrypoints
  reconstruct_analysis.py
```

---

## Dependencies

- Python **3.11**.
- Package manager: **uv** (recommended) or **pip**/**poetry**. The project is
  installable as `xling-tok` (`pyproject.toml`).
- Pinned floor versions live in `requirements.txt`.
- For an exact lock:
  ```bash
  uv pip compile pyproject.toml -o requirements.lock
  ```

Optional extras:

| Extra        | Why you'd want it                                  |
|--------------|----------------------------------------------------|
| `hf`         | local Llama / Qwen via `transformers` + `torch`    |
| `api`        | OpenAI-compatible hosted backend                   |
| `notebook`   | JupyterLab + ipykernel                             |
| `dev`        | pytest, ruff, mypy                                 |

---

## Backends

Selectable per-config under `backend:`:

```yaml
backend:
  kind: dummy                      # offline default
  # kind: hf_local
  # model_id: meta-llama/Meta-Llama-3.1-8B-Instruct
  # dtype: bfloat16
  # ---
  # kind: openai
  # model: gpt-4o-mini
  # ---
  # kind: together                 # uses OpenAI-compatible client
  # base_url: https://api.together.xyz/v1
  # model: meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo
```

Missing API keys or extras are detected at construction time; the registry
falls back to the `dummy` backend with a logged warning so the rest of the
pipeline still produces artifacts.

---

## Tokenizers

Available presets (see `src/tokenizers/registry.py`):

- `tiktoken_cl100k`, `tiktoken_o200k`
- `llama3` → `meta-llama/Meta-Llama-3.1-8B-Instruct`
- `qwen25` → `Qwen/Qwen2.5-7B-Instruct`
- `byte`, `char`
- Custom: any `{kind: hf, hf_id: ...}` or `{kind: sentencepiece, model_path: ...}`

Train a SentencePiece tokenizer on the bilingual corpus:

```bash
python -m src.tokenizers.train_sentencepiece \
  --input  data/processed/bilingual.txt \
  --model_prefix models/sp_bpe \
  --vocab_size 32000 \
  --model_type bpe
```

---

## Data

`load_bilingual(...)` (in `src.data`) wraps:

- **FLORES-200** (`facebook/flores`)
- **XNLI** (`xnli`)
- **XQuAD** (`xquad`)
- **MGSM** (`juletxara/mgsm`)
- **XLSUM** (`csebuetnlp/xlsum`, optional)

The raw HF rows are normalized to the schema:

```
id, dataset, split, domain, language, text, paired_id
```

`paired_id` joins the EN and ZH rows of the same example. Misalignment fails
loudly via `assert_pair_ids_match`. All loaders cache to
`data/raw/<dataset>__<langs>__<split>.jsonl`. Normalization knobs are
documented in `src/data/normalize.py`.

---

## Experiments

| # | Script                          | Outputs                                         |
|---|---------------------------------|-------------------------------------------------|
| 1 | `run_tokenization_audit`        | per-sentence + per-tokenizer CSVs, bar/box/scatter, summary.md |
| 2 | `run_fixed_context`             | budget curves, examples-fit, cost-per-correct   |
| 3 | `run_radical_sensitivity`       | condition table + logit coefficients + plot     |
| 4 | `run_paraphrase_perturb`        | per-transform stability and Δ-accuracy summary  |
| 5 | `run_tokenizer_adaptation`      | counterfactual audit + optional retraining curves |

See `reports/methods.md` for the full design.

---

## Reproducibility

Every CLI invocation creates a timestamped run directory:

```
results/<experiment>/runs/<YYYYMMDD-HHMMSS>__<name>/
  config.yaml         # exact config used
  manifest.json       # seed, git commit, env, elapsed
  git_commit.txt
  run.log
```

The artifacts that downstream consumers care about (CSVs, plots, summary.md)
also live one level up at `results/<experiment>/` so the runs directory can be
treated as historical record without breaking the analysis.

To regenerate `reports/final_report.md` and the headline plots from CSVs alone
(no re-running of experiments):

```bash
python -m src.reconstruct_analysis
```

---

## Tests

```bash
pytest -q
```

The smoke tests cover the byte/char tokenizers, normalization, Experiment 1's
table builders, and the stats helpers — none of them require the network or
heavyweight model downloads.

---

## Interpretation guardrails

- A high TP for a language **does not** mean that language is intrinsically
  worse. It means the *tokenizer* fragments it more.
- Always separate four axes: token compression, context-window consequences,
  downstream task performance, and training-data exposure.
- Label every result as **tokenizer-specific**, **model-specific**, or
  **task-specific**.

---

## License

MIT.
