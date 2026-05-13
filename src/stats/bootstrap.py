"""Non-parametric bootstrap confidence intervals."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import NamedTuple

import numpy as np


class CI(NamedTuple):
    point: float
    low: float
    high: float
    n_boot: int
    alpha: float


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> CI:
    """Percentile bootstrap CI for a one-sample statistic."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return CI(float("nan"), float("nan"), float("nan"), n_boot, alpha)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    samples = np.array([statistic(arr[i]) for i in idx])
    low, high = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return CI(float(statistic(arr)), float(low), float(high), n_boot, alpha)


def paired_bootstrap_ci(
    a: Sequence[float],
    b: Sequence[float],
    statistic: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> CI:
    """Paired bootstrap CI for a statistic of two aligned series.

    Resamples indices jointly so paired structure is preserved.
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.size != b_arr.size:
        raise ValueError(f"Length mismatch: {a_arr.size} vs {b_arr.size}")
    if a_arr.size == 0:
        return CI(float("nan"), float("nan"), float("nan"), n_boot, alpha)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a_arr.size, size=(n_boot, a_arr.size))
    samples = np.array([statistic(a_arr[i], b_arr[i]) for i in idx])
    low, high = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return CI(float(statistic(a_arr, b_arr)), float(low), float(high), n_boot, alpha)
