"""Tokenizer abstraction layer.

A `TokenizerSpec` exposes a uniform `encode(text) -> list[int]` interface plus
metadata (vocab size, family, version). Backends supported:

    - tiktoken (cl100k_base, o200k_base if available)  -> GPT-family proxies
    - HuggingFace tokenizers AutoTokenizer             -> Llama-3, Qwen, etc.
    - SentencePiece (BPE / Unigram), trained or local model
    - byte baseline (UTF-8 bytes)
    - char baseline (Unicode codepoints)

The registry is small, transparent, and degrades gracefully when an optional
backend isn't installed (raises a clear ImportError at construction time).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class TokenizerSpec:
    """Uniform tokenizer wrapper used everywhere downstream."""

    name: str
    family: str  # "tiktoken" | "hf" | "sentencepiece" | "byte" | "char"
    encode_fn: Callable[[str], list[int]]
    vocab_size: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def encode(self, text: str) -> list[int]:
        return self.encode_fn(text)

    def count(self, text: str) -> int:
        return len(self.encode_fn(text))


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _build_tiktoken(name: str, encoding: str) -> TokenizerSpec:
    try:
        import tiktoken
    except ImportError as e:
        raise ImportError("tiktoken not installed; pip install tiktoken") from e
    enc = tiktoken.get_encoding(encoding)
    return TokenizerSpec(
        name=name,
        family="tiktoken",
        encode_fn=lambda t: enc.encode(t, disallowed_special=()),
        vocab_size=enc.n_vocab,
        metadata={"encoding": encoding},
    )


def _build_hf(name: str, hf_id: str, revision: str | None = None) -> TokenizerSpec:
    try:
        from tokenizers import Tokenizer  # noqa: F401
        from transformers import AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "transformers + tokenizers required for HF tokenizers; "
            "pip install 'transformers>=4.44' tokenizers"
        ) from e
    tok = AutoTokenizer.from_pretrained(hf_id, revision=revision, use_fast=True)
    vocab = getattr(tok, "vocab_size", None)
    return TokenizerSpec(
        name=name,
        family="hf",
        encode_fn=lambda t: tok.encode(t, add_special_tokens=False),
        vocab_size=vocab,
        metadata={"hf_id": hf_id, "revision": revision},
    )


def _build_sentencepiece(name: str, model_path: str | Path, family_hint: str) -> TokenizerSpec:
    try:
        import sentencepiece as spm
    except ImportError as e:
        raise ImportError("sentencepiece not installed; pip install sentencepiece") from e
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"SentencePiece model not found at {model_path}. "
            "Train one with src/tokenizers/train_sentencepiece.py first."
        )
    sp = spm.SentencePieceProcessor()
    sp.Load(str(model_path))
    return TokenizerSpec(
        name=name,
        family="sentencepiece",
        encode_fn=lambda t: sp.encode(t, out_type=int),
        vocab_size=sp.get_piece_size(),
        metadata={"model_path": str(model_path), "model_type": family_hint},
    )


def _build_byte(name: str = "byte") -> TokenizerSpec:
    return TokenizerSpec(
        name=name,
        family="byte",
        encode_fn=lambda t: list(t.encode("utf-8")),
        vocab_size=256,
        metadata={"note": "raw UTF-8 bytes; lower bound on length"},
    )


def _build_char(name: str = "char") -> TokenizerSpec:
    return TokenizerSpec(
        name=name,
        family="char",
        encode_fn=lambda t: [ord(c) for c in t],
        vocab_size=None,
        metadata={"note": "Unicode codepoints; one token per character"},
    )


# ---------------------------------------------------------------------------
# Registry-style entrypoint
# ---------------------------------------------------------------------------

# Preset names the configs can refer to.
_PRESETS: dict[str, dict[str, Any]] = {
    "tiktoken_cl100k": {"kind": "tiktoken", "encoding": "cl100k_base"},
    "tiktoken_o200k":  {"kind": "tiktoken", "encoding": "o200k_base"},
    "llama3":          {"kind": "hf", "hf_id": "meta-llama/Meta-Llama-3.1-8B-Instruct"},
    "qwen25":          {"kind": "hf", "hf_id": "Qwen/Qwen2.5-7B-Instruct"},
    "byte":            {"kind": "byte"},
    "char":            {"kind": "char"},
}


def list_tokenizers() -> list[str]:
    return sorted(_PRESETS.keys())


def build_tokenizer(spec: str | dict[str, Any]) -> TokenizerSpec:
    """Build from a preset name or an explicit dict.

    Examples:
        build_tokenizer("llama3")
        build_tokenizer({"kind": "hf", "hf_id": "Qwen/Qwen2.5-7B-Instruct", "name": "qwen"})
        build_tokenizer({"kind": "sentencepiece", "model_path": "models/sp_bpe.model",
                         "family_hint": "bpe", "name": "sp_bpe_zh"})
    """
    if isinstance(spec, str):
        if spec not in _PRESETS:
            raise KeyError(f"Unknown tokenizer preset '{spec}'. Known: {list_tokenizers()}")
        s: dict[str, Any] = {"name": spec, **_PRESETS[spec]}
    else:
        s = dict(spec)
    name = s.get("name") or s.get("kind", "tokenizer")
    kind = s.get("kind")
    if kind == "tiktoken":
        return _build_tiktoken(name, s["encoding"])
    if kind == "hf":
        return _build_hf(name, s["hf_id"], s.get("revision"))
    if kind == "sentencepiece":
        return _build_sentencepiece(
            name, s["model_path"], s.get("family_hint", "bpe")
        )
    if kind == "byte":
        return _build_byte(name)
    if kind == "char":
        return _build_char(name)
    raise ValueError(f"Unknown tokenizer kind: {kind!r}")
