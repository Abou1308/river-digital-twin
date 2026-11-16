"""Drift detection on incoming lab results.

Every lab result is compared with the soft sensor's prediction made *just before* it arrived.
The standardised innovation  z = (ln observed - mean_log) / sd_log  should be ~N(0, 1) if the model
is well calibrated; persistent bias or excess spread signals drift.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import defaultdict, deque

from rdt.softsensor.estimate import MetalEstimate
from rdt.state import Alert


def standardised_innovation(prior: MetalEstimate, metal: str, observed: float) -> float:
    i = prior.metals.index(metal)
    sd = math.sqrt(prior.cov_log[i, i])
    return (math.log(observed) - prior.mean_log[i]) / sd


class DriftDetector(ABC):
    @abstractmethod
    def observe(self, prior: MetalEstimate, metal: str, observed: float, censored: bool) -> Alert | None:
        """Process one lab result; return an Alert if drift is detected."""


class IntervalExceedanceDetector(DriftDetector):
    """Simple placeholder: alert when `k` consecutive results fall outside the central `level` interval.

    TODO: replace / complement with a CUSUM or Page-Hinkley test on standardised innovations,
    which detects gradual bias rather than only large misses.
    """

    def __init__(self, k: int = 3, level: float = 0.9) -> None:
        from statistics import NormalDist

        self.k = k
        self.z = NormalDist().inv_cdf(0.5 + level / 2)
        self._recent: dict[tuple[str, str], deque[bool]] = defaultdict(lambda: deque(maxlen=k))

    def observe(self, prior: MetalEstimate, metal: str, observed: float, censored: bool) -> Alert | None:
        if censored or observed <= 0:
            return None
        z = standardised_innovation(prior, metal, observed)
        window = self._recent[(prior.station, metal)]
        window.append(abs(z) > self.z)
        if len(window) == self.k and all(window):
            window.clear()
            return Alert(
                time_utc=prior.time_utc,
                station=prior.station,
                metal=metal,
                kind="drift",
                severity="warning",
                message=f"{metal.upper()} outside prediction interval {self.k}x in a row (last z={z:+.1f})",
            )
        return None
