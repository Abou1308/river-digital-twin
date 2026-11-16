"""Evaluation metrics shared by offline experiments and the live twin."""

from __future__ import annotations

import numpy as np


def median_abs_pct_error(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Median of |pred - obs| / obs, in percent (the metric used in the exploration)."""
    observed, predicted = np.asarray(observed, float), np.asarray(predicted, float)
    return float(np.nanmedian(np.abs(predicted - observed) / observed) * 100)


def interval_coverage(observed: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of observations inside [lower, upper]; should match the nominal level (e.g. 0.9)."""
    observed = np.asarray(observed, float)
    return float(np.mean((observed >= np.asarray(lower)) & (observed <= np.asarray(upper))))


# TODO: class-level metrics once quality.classify is implemented, e.g. accuracy of the most likely
# class, Brier score / log loss of the class probabilities, reliability (calibration) curves.


# --- significance of differences between models (autocorrelation-aware) ----------------------------

def diebold_mariano(loss_diff: np.ndarray, lag: int) -> tuple[float, float]:
    """Diebold-Mariano test of mean(loss_diff) = 0 with Newey-West (HAC) variance and the
    Harvey-Leybourne-Newbold small-sample correction. Returns (statistic, two-sided p)."""
    from scipy import stats

    d = np.asarray(loss_diff, float)
    d = d[np.isfinite(d)]
    n = len(d)
    dc = d - d.mean()
    gamma = [np.dot(dc[: n - j], dc[j:]) / n for j in range(lag + 1)]
    lrv = gamma[0] + 2 * sum((1 - j / (lag + 1)) * gamma[j] for j in range(1, lag + 1))
    if lrv <= 0:
        lrv = gamma[0]
    h = lag + 1
    stat = d.mean() / np.sqrt(lrv / n) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(stat), float(2 * stats.t.sf(abs(stat), n - 1))


def block_bootstrap_median_diff(a: np.ndarray, b: np.ndarray, block: int, n_boot: int = 2000,
                                seed: int = 0) -> tuple[float, float]:
    """95% circular block-bootstrap interval of median(a) - median(b) for paired, time-ordered errors."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    n = len(a)
    nb = int(np.ceil(n / block))
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = (rng.integers(0, n, nb)[:, None] + np.arange(block)[None, :]).ravel()[:n] % n
        diffs[i] = np.median(a[idx]) - np.median(b[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


def holm_count(pvalues: np.ndarray, alpha: float = 0.05) -> int:
    """Number of hypotheses rejected by the Holm step-down procedure."""
    p = np.sort(np.asarray(pvalues, float))
    return int((p < alpha / (len(p) - np.arange(len(p)))).cumprod().sum())
