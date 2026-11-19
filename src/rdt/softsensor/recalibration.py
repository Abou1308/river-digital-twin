"""Online recalibration of prediction ranges: a wrapper around any soft sensor.

The Kalman soft sensor widens its ranges per metal by a factor fitted once on training, so that 90% of
training lab results fall inside the 90% range. The calibration check (analysis/06_class_calibration.py)
showed that on 2021-2024 some ranges are still too narrow (Cu and Pb at Eijsden) and some too wide
(Cu at Lobith). This wrapper keeps re-estimating that factor from the lab results that have arrived:

    factor^2 = (prior_n * 1 + n * (q90(|z|) / 1.645)^2) / (prior_n + n)

where z are the wrapped model's standardised errors for the n results so far at that station and metal,
q90 is their 90th percentile, and `prior_n` is the weight, in lab results, given to the training
calibration (factor 1). The factor can widen or narrow the range; predictions themselves are unchanged.

Leakage: a result's z is only used for predictions made after the result arrived (`available_utc`),
the same rule the soft sensors apply to the results themselves.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import datetime

import numpy as np
import pandas as pd

from rdt.softsensor.base import LabObservation, SoftSensor
from rdt.softsensor.estimate import MetalEstimate

_Z90 = 1.6448536269514722


class RecalibratedSoftSensor(SoftSensor):
    def __init__(self, inner: SoftSensor, prior_n: float) -> None:
        super().__init__(inner.metals, inner.features)
        self.inner = inner
        self.prior_n = prior_n
        self.name = inner.name
        # {(station, metal): [(available time, |z|)]}, in arrival order
        self._errors: dict[tuple[str, str], list[tuple[pd.Timestamp, float]]] = {}

    def fit(self, train: pd.DataFrame, cutoff: pd.Timestamp) -> None:
        self.inner.fit(train, cutoff)

    def assimilate(self, station: str, t: datetime, metal: str, value: float, censored: bool,
                   features: Mapping[str, float] | None = None, available_utc: datetime | None = None) -> None:
        if metal in self.metals and not censored and np.isfinite(value) and value > 0:
            # The error of the prediction the twin had for the sample time, before this result was known.
            prior = self.inner.predict(station, t, features or {}, as_of=t)
            i = prior.metals.index(metal)
            z = (math.log(value) - prior.mean_log[i]) / math.sqrt(prior.cov_log[i, i])
            available = pd.Timestamp(available_utc if available_utc is not None else t)
            self._errors.setdefault((station, metal), []).append((available, abs(z)))
        self.inner.assimilate(station, t, metal, value, censored, features, available_utc)

    def factor(self, station: str, metal: str, as_of: datetime | None = None) -> float:
        """Range factor from the results that had arrived strictly before `as_of` (all if None)."""
        errors = self._errors.get((station, metal), [])
        if as_of is not None:
            cut = pd.Timestamp(as_of)
            errors = [e for e in errors if e[0] < cut]
        if not errors:
            return 1.0
        empirical = np.quantile([z for _, z in errors], 0.9) / _Z90
        n = len(errors)
        return math.sqrt((self.prior_n + n * empirical**2) / (self.prior_n + n))

    def predict(self, station: str, t: datetime, features: Mapping[str, float],
                as_of: datetime | None = None) -> MetalEstimate:
        est = self.inner.predict(station, t, features, as_of=as_of)
        f = np.array([self.factor(station, m, as_of) for m in est.metals])
        return MetalEstimate(est.station, est.time_utc, est.metals, est.mean_log,
                             est.cov_log * np.outer(f, f), model=est.model)

    def known(self, station: str, as_of: datetime | None) -> list[LabObservation]:
        return self.inner.known(station, as_of)

    def assimilate_property(self, station: str, t: datetime, prop: str, value: float, censored: bool) -> None:
        self.inner.assimilate_property(station, t, prop, value, censored)

    def sediment_properties(self, station: str) -> dict[str, tuple[datetime, float]]:
        return self.inner.sediment_properties(station)
