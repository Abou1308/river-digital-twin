"""Calibration checks: do predicted probabilities and prediction ranges come true as often as they claim?

Used by analysis/06_class_calibration.py. Two kinds of check:
  * probability forecasts of a yes/no event (e.g. "at least moderately contaminated"): reliability table
    and the Murphy decomposition of the Brier score into reliability (miscalibration), resolution and
    uncertainty;
  * log-normal concentration forecasts: the probability integral transform (PIT), i.e. where each lab
    value falls in its predicted distribution. Calibrated forecasts give PIT values uniform on [0, 1].
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

_Z90 = NormalDist().inv_cdf(0.95)


def _bin_index(p: np.ndarray, n_bins: int) -> np.ndarray:
    """Equal-width bins on [0, 1]; p = 1 goes in the last bin."""
    return np.minimum((p * n_bins).astype(int), n_bins - 1)


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a proportion k / n."""
    if n == 0:
        return float("nan"), float("nan")
    phat = k / n
    centre = (phat + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return float(centre - half), float(centre + half)


def reliability_table(p: np.ndarray, outcome: np.ndarray, n_bins: int = 5) -> pd.DataFrame:
    """Per probability bin: number of forecasts, mean forecast probability, observed frequency of the
    event and its 95% Wilson interval. The interval assumes independent samples, so it is too narrow
    for autocorrelated series."""
    p, o = np.asarray(p, float), np.asarray(outcome, bool)
    idx = _bin_index(p, n_bins)
    rows = []
    for b in range(n_bins):
        sel = idx == b
        n, k = int(sel.sum()), int(o[sel].sum())
        lo, hi = _wilson(k, n)
        rows.append({"bin": f"{b / n_bins:.1f}-{(b + 1) / n_bins:.1f}", "n": n,
                     "mean_forecast": float(p[sel].mean()) if n else float("nan"),
                     "observed": k / n if n else float("nan"), "observed_lo": lo, "observed_hi": hi})
    return pd.DataFrame(rows)


def brier_decomposition(p: np.ndarray, outcome: np.ndarray, n_bins: int = 5) -> dict[str, float]:
    """Brier score of a yes/no forecast and its Murphy decomposition over probability bins:
    brier ~ reliability - resolution + uncertainty. Reliability is 0 for perfect calibration (lower is
    better); resolution is how far the bins' outcomes differ from the overall rate (higher is better)."""
    p, o = np.asarray(p, float), np.asarray(outcome, float)
    idx = _bin_index(p, n_bins)
    base = o.mean()
    rel = res = 0.0
    for b in np.unique(idx):
        sel = idx == b
        w = sel.mean()
        rel += w * (p[sel].mean() - o[sel].mean()) ** 2
        res += w * (o[sel].mean() - base) ** 2
    return {"brier": float(np.mean((p - o) ** 2)), "reliability": float(rel), "resolution": float(res),
            "uncertainty": float(base * (1 - base))}


def block_bootstrap_mean(x: np.ndarray, block: int, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% circular block-bootstrap interval of mean(x) for a time-ordered series."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    n = len(x)
    nb = int(np.ceil(n / block))
    idx = (rng.integers(0, n, (n_boot, nb))[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n] % n
    lo, hi = np.percentile(x[idx].mean(axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def pit_lognormal(observed: np.ndarray, median: np.ndarray, upper90: np.ndarray) -> np.ndarray:
    """PIT of observations under log-normal forecasts given by their median and the upper end of the
    central 90% range (log-sd = ln(upper90 / median) / z_0.95)."""
    sd = np.log(np.asarray(upper90, float) / np.asarray(median, float)) / _Z90
    z = np.log(np.asarray(observed, float) / np.asarray(median, float)) / sd
    return np.array([NormalDist().cdf(v) for v in z])


def pit_coverage(pit: np.ndarray, levels: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95)) -> dict[float, float]:
    """Share of observations inside each central range; equal to the level when calibrated."""
    pit = np.asarray(pit, float)
    return {lv: float(np.mean(np.abs(pit - 0.5) <= lv / 2)) for lv in levels}
