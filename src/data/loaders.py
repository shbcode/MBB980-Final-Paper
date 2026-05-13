"""Bilingual corpus loaders.

For FLORES-200 we deliberately bypass the legacy HuggingFace script
(`facebook/flores`) because it does ``open(filename, "r")`` without
``encoding="utf-8"``, which on Windows falls back to cp1252 and fails for
any non-Latin script. Instead we download the official tarball directly,
extract once into ``data/raw/flores200/``, and read the per-language TSVs
with explicit UTF-8.

XNLI / XQuAD / MGSM / XLSUM are still loaded via `datasets` because their
loaders are parquet/Arrow-based and do not have the encoding bug.

Every loader returns a list[BilingualPair] with stable `paired_id`s so later
code can verify alignment with `assert_pair_ids_match`.
"""

from __future__ import annotations

import csv
import shutil
import tarfile
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from ..utils.io import read_jsonl, write_jsonl
from ..utils.logging import get_logger
from .normalize import NormalizationOptions, normalize_text

log = get_logger(__name__)


# --- ISO/HF code mappings -------------------------------------------------

# FLORES-200 uses BCP-47-ish codes with script suffix.
FLORES_CODES = {
    "en": "eng_Latn",
    "zh": "zho_Hans",
    "zh_t": "zho_Hant",
    "ja": "jpn_Jpan",
    "ar": "arb_Arab",
    "hi": "hin_Deva",
    "ta": "tam_Taml",
}

# Two-letter language codes used by XNLI / XQuAD.
XNLI_CODES = {"en": "en", "zh": "zh"}
XQUAD_CODES = {"en": "xquad.en", "zh": "xquad.zh"}
MGSM_CODES = {"en": "en", "zh": "zh"}


@dataclass
class BilingualPair:
    """A single aligned EN<->ZH (or other) example."""

    id: str            # globally unique row id
    dataset: str       # e.g. "flores200", "xnli"
    split: str         # "dev" / "devtest" / "test" / ...
    domain: str        # corpus subdomain or "general"
    language: str      # ISO 639-1: "en", "zh", ...
    text: str          # actual sentence
    paired_id: str     # join key linking EN <-> ZH rows


def _cache_path(cache_dir: Path, dataset: str, split: str, lang_a: str, lang_b: str) -> Path:
    return cache_dir / f"{dataset}__{lang_a}-{lang_b}__{split}.jsonl"


def _maybe_load_cache(path: Path) -> list[BilingualPair] | None:
    if not path.exists():
        return None
    rows = read_jsonl(path)
    return [BilingualPair(**r) for r in rows]


def _save_cache(rows: Iterable[BilingualPair], path: Path) -> None:
    write_jsonl([asdict(r) for r in rows], path)


def _normalize_pairs(
    pairs: list[BilingualPair], opts: NormalizationOptions | None
) -> list[BilingualPair]:
    if opts is None:
        return pairs
    out: list[BilingualPair] = []
    for p in pairs:
        out.append(
            BilingualPair(
                id=p.id,
                dataset=p.dataset,
                split=p.split,
                domain=p.domain,
                language=p.language,
                text=normalize_text(p.text, opts),
                paired_id=p.paired_id,
            )
        )
    return out


# --- FLORES-200 ----------------------------------------------------------

# The official Meta-AI tarball. The flores.py script in `datasets` points to
# the same tinyurl, but its file-reading code is broken on Windows (cp1252).
FLORES_TARBALL_URLS = (
    "https://tinyurl.com/flores200dataset",
    "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz",
)


def _flores_root(cache_dir: Path) -> Path:
    """Project-local FLORES root: data/raw/flores200/."""
    return cache_dir / "flores200"


def _ensure_flores_extracted(cache_dir: Path) -> Path:
    """Return path to the extracted `flores200_dataset/` directory.

    Resolution order, in priority:
      1. ``data/raw/flores200/flores200_dataset`` (project-local, preferred).
      2. The HuggingFace cache extraction directory (we reuse what `datasets`
         already downloaded, when present, so we don't redownload 25 MB).
      3. Download the tarball ourselves, extract under ``data/raw/flores200/``.
    """
    project_dir = _flores_root(cache_dir) / "flores200_dataset"
    if (project_dir / "devtest" / "eng_Latn.devtest").exists():
        return project_dir

    # Reuse HF's already-extracted copy if present (zero-cost on systems where
    # `datasets` previously fetched it before we noticed the bug).
    hf_cache = Path.home() / ".cache" / "huggingface" / "datasets" / "downloads" / "extracted"
    if hf_cache.exists():
        for sub in hf_cache.iterdir():
            cand = sub / "flores200_dataset"
            if (cand / "devtest" / "eng_Latn.devtest").exists():
                log.info("Reusing FLORES extraction from HF cache: %s", cand)
                return cand

    # Fall back to a fresh download into our project cache.
    target_dir = _flores_root(cache_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    tarball = target_dir / "flores200_dataset.tar.gz"
    if not tarball.exists():
        last_err: Exception | None = None
        for url in FLORES_TARBALL_URLS:
            try:
                log.info("Downloading FLORES-200 tarball from %s", url)
                with urllib.request.urlopen(url) as resp, tarball.open("wb") as out:
                    shutil.copyfileobj(resp, out)
                break
            except Exception as e:
                log.warning("FLORES download from %s failed: %s", url, e)
                last_err = e
        else:
            raise RuntimeError(
                f"Could not download FLORES-200 from any of {FLORES_TARBALL_URLS}"
            ) from last_err

    log.info("Extracting FLORES-200 tarball -> %s", target_dir)
    with tarfile.open(tarball, "r:gz") as tf:
        tf.extractall(target_dir)
    return target_dir / "flores200_dataset"


def _read_flores_lang(root: Path, code: str, split: str) -> list[str]:
    """Read a single per-language TSV with explicit UTF-8 (the bug-fix part)."""
    path = root / split / f"{code}.{split}"
    if not path.exists():
        raise FileNotFoundError(
            f"FLORES file not found: {path}. The tarball may have a different layout."
        )
    with path.open("r", encoding="utf-8", newline="") as f:
        return [line.rstrip("\n") for line in f]


def _read_flores_metadata(root: Path, split: str) -> list[dict[str, str]]:
    """Read ``metadata_<split>.tsv``; one entry per line, aligned with the
    sentence files. Columns include URL, domain, topic, has_image, has_hyperlink.
    """
    path = root / f"metadata_{split}.tsv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def load_flores(
    split: str = "devtest",
    languages: tuple[str, str] = ("en", "zh"),
    cache_dir: str | Path = "data/raw",
    max_examples: int | None = None,
) -> list[BilingualPair]:
    """Load FLORES-200 parallel sentences for two languages.

    Returns one BilingualPair per (language, sentence). Pairs across languages
    share the same `paired_id` (the FLORES line index).

    Implementation note: we read the official Meta tarball directly to avoid
    the cp1252-on-Windows decode bug in the legacy `datasets` script.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = _cache_path(cache_dir, "flores200", split, *languages)
    cached = _maybe_load_cache(cache)
    if cached is not None:
        log.info("FLORES cache hit: %s (%d rows)", cache, len(cached))
        return cached if max_examples is None else _truncate_pairs(cached, max_examples)

    flores_root = _ensure_flores_extracted(cache_dir)
    metadata = _read_flores_metadata(flores_root, split)

    out: list[BilingualPair] = []
    expected_n: int | None = None
    for lang in languages:
        if lang not in FLORES_CODES:
            raise KeyError(f"Unknown FLORES language: {lang}")
        code = FLORES_CODES[lang]
        sentences = _read_flores_lang(flores_root, code, split)
        if expected_n is None:
            expected_n = len(sentences)
        elif len(sentences) != expected_n:
            raise ValueError(
                f"FLORES misalignment: {lang} has {len(sentences)} lines, "
                f"expected {expected_n}"
            )
        for i, text in enumerate(sentences):
            md = metadata[i] if i < len(metadata) else {}
            domain = md.get("domain") or "general"
            out.append(
                BilingualPair(
                    id=f"flores200:{split}:{lang}:{i}",
                    dataset="flores200",
                    split=split,
                    domain=domain,
                    language=lang,
                    text=text,
                    paired_id=f"flores200:{split}:{i}",
                )
            )
    _save_cache(out, cache)
    log.info("FLORES cached -> %s (%d rows)", cache, len(out))
    return out if max_examples is None else _truncate_pairs(out, max_examples)


def _truncate_pairs(pairs: list[BilingualPair], n: int) -> list[BilingualPair]:
    """Keep the first n distinct paired_ids (so EN/ZH stay aligned)."""
    keep_ids: list[str] = []
    seen: set[str] = set()
    for p in pairs:
        if p.paired_id not in seen:
            seen.add(p.paired_id)
            keep_ids.append(p.paired_id)
            if len(keep_ids) >= n:
                break
    keep_set = set(keep_ids)
    return [p for p in pairs if p.paired_id in keep_set]


# --- XNLI / XQuAD / MGSM (lighter wrappers) ------------------------------

def load_xnli(
    split: str = "validation",
    languages: tuple[str, str] = ("en", "zh"),
    cache_dir: str | Path = "data/raw",
    max_examples: int | None = None,
) -> list[BilingualPair]:
    """XNLI: use the `premise` field; `paired_id` is the row index (rows aligned across langs)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = _cache_path(cache_dir, "xnli", split, *languages)
    cached = _maybe_load_cache(cache)
    if cached is not None:
        return cached if max_examples is None else _truncate_pairs(cached, max_examples)

    from datasets import load_dataset

    out: list[BilingualPair] = []
    for lang in languages:
        ds = load_dataset("xnli", XNLI_CODES[lang], split=split)
        for i, row in enumerate(ds):
            out.append(
                BilingualPair(
                    id=f"xnli:{split}:{lang}:{i}",
                    dataset="xnli",
                    split=split,
                    domain="nli",
                    language=lang,
                    text=row["premise"],
                    paired_id=f"xnli:{split}:{i}",
                )
            )
    _save_cache(out, cache)
    return out if max_examples is None else _truncate_pairs(out, max_examples)


def load_xquad(
    languages: tuple[str, str] = ("en", "zh"),
    cache_dir: str | Path = "data/raw",
    max_examples: int | None = None,
) -> list[BilingualPair]:
    """XQuAD: use context as the audited text; paired_id = row index."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = _cache_path(cache_dir, "xquad", "validation", *languages)
    cached = _maybe_load_cache(cache)
    if cached is not None:
        return cached if max_examples is None else _truncate_pairs(cached, max_examples)

    from datasets import load_dataset

    out: list[BilingualPair] = []
    for lang in languages:
        ds = load_dataset("xquad", XQUAD_CODES[lang], split="validation")
        for i, row in enumerate(ds):
            out.append(
                BilingualPair(
                    id=f"xquad:val:{lang}:{i}",
                    dataset="xquad",
                    split="validation",
                    domain="qa",
                    language=lang,
                    text=row["context"],
                    paired_id=f"xquad:val:{i}",
                )
            )
    _save_cache(out, cache)
    return out if max_examples is None else _truncate_pairs(out, max_examples)


def load_mgsm(
    languages: tuple[str, str] = ("en", "zh"),
    cache_dir: str | Path = "data/raw",
    max_examples: int | None = None,
) -> list[BilingualPair]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = _cache_path(cache_dir, "mgsm", "test", *languages)
    cached = _maybe_load_cache(cache)
    if cached is not None:
        return cached if max_examples is None else _truncate_pairs(cached, max_examples)

    from datasets import load_dataset

    out: list[BilingualPair] = []
    for lang in languages:
        ds = load_dataset("juletxara/mgsm", MGSM_CODES[lang], split="test")
        for i, row in enumerate(ds):
            out.append(
                BilingualPair(
                    id=f"mgsm:test:{lang}:{i}",
                    dataset="mgsm",
                    split="test",
                    domain="math",
                    language=lang,
                    text=row["question"],
                    paired_id=f"mgsm:test:{i}",
                )
            )
    _save_cache(out, cache)
    return out if max_examples is None else _truncate_pairs(out, max_examples)


# --- High-level orchestrator ---------------------------------------------

def load_bilingual(
    dataset: str,
    *,
    split: str | None = None,
    languages: tuple[str, str] = ("en", "zh"),
    cache_dir: str | Path = "data/raw",
    max_examples: int | None = None,
    normalization: NormalizationOptions | None = None,
    save_normalized_to: str | Path | None = None,
) -> list[BilingualPair]:
    """Top-level loader; routes to the right backend and applies normalization.

    Always validates EN<->ZH paired_id alignment and fails loudly if a side
    is missing. Optionally writes the normalized rows to `data/processed/...`.
    """
    name = dataset.lower()
    if name == "flores200":
        pairs = load_flores(split or "devtest", languages, cache_dir, max_examples)
    elif name == "xnli":
        pairs = load_xnli(split or "validation", languages, cache_dir, max_examples)
    elif name == "xquad":
        pairs = load_xquad(languages, cache_dir, max_examples)
    elif name == "mgsm":
        pairs = load_mgsm(languages, cache_dir, max_examples)
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")

    assert_pair_ids_match(pairs, languages)
    pairs = _normalize_pairs(pairs, normalization)

    if save_normalized_to:
        _save_cache(pairs, Path(save_normalized_to))

    return pairs


def assert_pair_ids_match(pairs: list[BilingualPair], languages: tuple[str, str]) -> None:
    """Loud failure if the two language sides aren't perfectly aligned by paired_id."""
    by_lang: dict[str, set[str]] = {}
    for p in pairs:
        by_lang.setdefault(p.language, set()).add(p.paired_id)
    if set(languages) - by_lang.keys():
        missing = set(languages) - by_lang.keys()
        raise ValueError(f"Missing language sides: {missing}")
    a, b = languages
    only_a = by_lang[a] - by_lang[b]
    only_b = by_lang[b] - by_lang[a]
    if only_a or only_b:
        raise ValueError(
            f"Bilingual misalignment: {len(only_a)} ids only in {a} "
            f"(e.g. {sorted(only_a)[:3]}); "
            f"{len(only_b)} ids only in {b} (e.g. {sorted(only_b)[:3]})."
        )
