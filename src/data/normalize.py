"""Text normalization utilities for cross-lingual fairness experiments.

Operations are intentionally orthogonal so callers can ablate each one:
    - Unicode NFKC normalization
    - Punctuation normalization (full-width <-> ASCII)
    - Simplified <-> Traditional Chinese (OpenCC)
    - Number-format normalization (full-width digits, Chinese numerals)

Every function is pure: text in, text out.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Full-width punctuation -> ASCII map (commonly produced by Chinese typing systems).
_FULLWIDTH_PUNCT_MAP = {
    "，": ",", "。": ".", "！": "!", "？": "?", "；": ";", "：": ":",
    "（": "(", "）": ")", "【": "[", "】": "]", "「": '"', "」": '"',
    "『": '"', "』": '"', "“": '"', "”": '"', "‘": "'", "’": "'",
    "—": "-", "·": ".", "／": "/", "％": "%",
}

_FULLWIDTH_DIGIT_MAP = {chr(0xFF10 + i): str(i) for i in range(10)}

_CN_NUMERAL_MAP = {
    "零": "0", "〇": "0", "一": "1", "二": "2", "两": "2", "三": "3",
    "四": "4", "五": "5", "六": "6", "七": "7", "八": "8", "九": "9",
}


@dataclass(frozen=True)
class NormalizationOptions:
    unicode_nfkc: bool = True
    punctuation: bool = True
    numbers: bool = False  # off by default; can change semantics
    simplify: bool = False  # convert traditional -> simplified
    traditionalize: bool = False  # convert simplified -> traditional
    collapse_whitespace: bool = True


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def normalize_punctuation(text: str) -> str:
    return text.translate(str.maketrans(_FULLWIDTH_PUNCT_MAP))


def normalize_numbers(text: str) -> str:
    text = text.translate(str.maketrans(_FULLWIDTH_DIGIT_MAP))
    text = text.translate(str.maketrans(_CN_NUMERAL_MAP))
    return text


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _opencc(direction: str) -> "OpenCC":  # type: ignore[name-defined]
    """Lazy OpenCC loader. direction in {'t2s.json', 's2t.json'}."""
    from opencc import OpenCC

    return OpenCC(direction.replace(".json", ""))


def t2s(text: str) -> str:
    """Traditional -> Simplified Chinese."""
    return _opencc("t2s.json").convert(text)


def s2t(text: str) -> str:
    """Simplified -> Traditional Chinese."""
    return _opencc("s2t.json").convert(text)


def normalize_text(text: str, opts: NormalizationOptions = NormalizationOptions()) -> str:
    if opts.unicode_nfkc:
        text = normalize_unicode(text)
    if opts.punctuation:
        text = normalize_punctuation(text)
    if opts.numbers:
        text = normalize_numbers(text)
    if opts.simplify and opts.traditionalize:
        raise ValueError("simplify and traditionalize are mutually exclusive")
    if opts.simplify:
        text = t2s(text)
    elif opts.traditionalize:
        text = s2t(text)
    if opts.collapse_whitespace:
        text = collapse_whitespace(text)
    return text
