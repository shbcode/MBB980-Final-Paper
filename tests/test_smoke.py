"""Smoke tests that work without HF model downloads or the network.

These are intentionally tiny: they verify the pipeline plumbing is wired
correctly. Real numbers come from running the experiments.
"""

from __future__ import annotations

from src.data.normalize import (
    NormalizationOptions,
    normalize_punctuation,
    normalize_text,
    normalize_unicode,
)
from src.experiments.tokenization_audit import per_sentence_table, summary_by_tokenizer
from src.data.loaders import BilingualPair
from src.stats.bootstrap import bootstrap_ci
from src.stats.tests import paired_wilcoxon
from src.tokenizers import build_tokenizer


def test_byte_and_char_tokenizers():
    byte = build_tokenizer("byte")
    ch = build_tokenizer("char")
    assert byte.count("hi") == 2
    assert byte.count("汉") == 3  # 3-byte UTF-8
    assert ch.count("hi") == 2
    assert ch.count("汉字") == 2


def test_normalize_pipeline():
    raw = "Hello， world！  42"
    out = normalize_text(raw, NormalizationOptions(numbers=False))
    assert "," in out and "!" in out
    assert normalize_unicode("ｈｅｌｌｏ") == "hello"
    assert normalize_punctuation("，。") == ",."


def test_audit_tables_with_byte_and_char():
    pairs = [
        BilingualPair("a:en:0", "demo", "test", "general", "en", "Hello world.", "p0"),
        BilingualPair("a:zh:0", "demo", "test", "general", "zh", "你好，世界。", "p0"),
        BilingualPair("a:en:1", "demo", "test", "general", "en", "Goodbye!", "p1"),
        BilingualPair("a:zh:1", "demo", "test", "general", "zh", "再见！", "p1"),
    ]
    toks = [build_tokenizer("byte"), build_tokenizer("char")]
    df = per_sentence_table(pairs, toks, ("en", "zh"))
    assert len(df) == 4  # 2 pairs * 2 tokenizers
    assert {"TP", "context_shrinkage"}.issubset(df.columns)

    summary = summary_by_tokenizer(df, ("en", "zh"), n_boot=50, seed=0)
    assert {"mean_TP", "wilcoxon_pvalue"}.issubset(summary.columns)
    assert (summary["n_pairs"] == 2).all()


def test_bootstrap_and_wilcoxon():
    ci = bootstrap_ci([1.0, 1.1, 0.9, 1.05, 0.95], n_boot=200, seed=0)
    assert ci.low <= ci.point <= ci.high
    res = paired_wilcoxon([2, 3, 4, 5], [1, 2, 3, 4])
    assert res.pvalue >= 0.0
    assert res.n == 4
