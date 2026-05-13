"""Hypothesis tests used in the analysis."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from scipy import stats


class TestResult(NamedTuple):
    statistic: float
    pvalue: float
    n: int
    test: str


def paired_wilcoxon(
    a: Sequence[float], b: Sequence[float], *, alternative: str = "two-sided"
) -> TestResult:
    """Wilcoxon signed-rank on paired samples (a - b)."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.size != b_arr.size:
        raise ValueError(f"Length mismatch: {a_arr.size} vs {b_arr.size}")
    diff = a_arr - b_arr
    nz = diff[diff != 0]
    if nz.size < 1:
        return TestResult(float("nan"), float("nan"), 0, "wilcoxon")
    res = stats.wilcoxon(a_arr, b_arr, alternative=alternative, zero_method="wilcox")
    return TestResult(float(res.statistic), float(res.pvalue), int(a_arr.size), "wilcoxon")
