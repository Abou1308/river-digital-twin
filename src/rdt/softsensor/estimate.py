"""The output of every soft sensor: a joint (log-normal) estimate of all metals at one station and time.

Working in natural-log space keeps concentrations positive, makes errors multiplicative (which is
how they behave in the data) and lets correlated metals (Cd/Pb/Zn) share one covariance matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from statistics import NormalDist

import numpy as np


@dataclass(frozen=True)
class MetalEstimate:
    station: str
    time_utc: datetime
    metals: tuple[str, ...]
    mean_log: np.ndarray        # shape (n_metals,), ln(mg/kg dw)
    cov_log: np.ndarray         # shape (n_metals, n_metals)
    model: str = ""

    def __post_init__(self) -> None:
        n = len(self.metals)
        if self.mean_log.shape != (n,) or self.cov_log.shape != (n, n):
            raise ValueError("mean_log / cov_log shapes do not match metals")

    def sd_log(self) -> dict[str, float]:
        return dict(zip(self.metals, np.sqrt(np.diag(self.cov_log)), strict=True))

    def median(self) -> dict[str, float]:
        """Median concentration (mg/kg dw) per metal."""
        return dict(zip(self.metals, np.exp(self.mean_log), strict=True))

    def interval(self, level: float = 0.9) -> tuple[dict[str, float], dict[str, float]]:
        """Central `level` prediction interval per metal, as (lower, upper) dicts in mg/kg dw."""
        z = NormalDist().inv_cdf(0.5 + level / 2)
        sd = np.sqrt(np.diag(self.cov_log))
        lo = np.exp(self.mean_log - z * sd)
        hi = np.exp(self.mean_log + z * sd)
        return dict(zip(self.metals, lo, strict=True)), dict(zip(self.metals, hi, strict=True))

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Draw `n` joint samples in mg/kg dw, shape (n, n_metals). Used for class probabilities."""
        rng = rng or np.random.default_rng()
        return np.exp(rng.multivariate_normal(self.mean_log, self.cov_log, size=n))
