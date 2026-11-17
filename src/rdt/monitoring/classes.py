"""Live class evaluation: the quality class of each lab sample against the class probabilities predicted
for its sample time.

A lab sample arrives as separate messages (one per metal, lutum and organic carbon) with the same
sample time. `ClassEvaluator.observe` collects them; once a sample is complete, the twin asks for the
prediction made for the sample time with only the results known then, and `ClassEvaluator.compare`
turns both into a `ClassComparison`.

The predicted probabilities use the sample's own lutum and organic matter, as does the observed class.
The comparison therefore measures how well the metal concentrations are predicted, not how well the
soil properties are; it matches the class check (V4) of the earlier MVP.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import numpy as np
import pandas as pd

from rdt.monitoring.evaluation import class_scores
from rdt.quality import QUALITY_CLASSES, sample_class, sample_class_probabilities
from rdt.softsensor.estimate import MetalEstimate
from rdt.state import ClassComparison

_PROPERTIES = ("lutum", "corg")


class ClassEvaluator:
    def __init__(
        self,
        metals: Sequence[str],
        limits: pd.DataFrame,
        soil_correction: pd.DataFrame,
        om_per_corg: float,
        n_draws: int = 5000,
        seed: int = 0,
    ) -> None:
        """`om_per_corg`: organic matter / organic carbon, since the lab reports carbon and the
        soil-type correction needs organic matter."""
        self.metals = tuple(metals)
        self.limits = limits
        self.soil_correction = soil_correction
        self.om_per_corg = om_per_corg
        self.n_draws = n_draws
        self._rng = np.random.default_rng(seed)
        self._required = set(self.metals) | set(_PROPERTIES)
        self._pending: dict[tuple[str, datetime], dict[str, float]] = {}

    def observe(self, station: str, t: datetime, key: str, value: float) -> dict[str, float] | None:
        """Add one lab value (a metal, 'lutum' or 'corg'); return the sample's values once it is complete.

        Censored values are used as reported: the few in the data are lutum / organic carbon below
        detection, which the correction clips up to its 2% minimum either way.
        """
        if key not in self._required:
            return None
        if not np.isfinite(value):
            raise ValueError(f"Lab value {key}={value} at {station} {t} is not a number.")
        sample = self._pending.setdefault((station, t), {})
        sample[key] = float(value)
        if not self._required <= sample.keys():
            return None
        return self._pending.pop((station, t))

    def compare(
        self,
        station: str,
        t: datetime,
        sample: dict[str, float],
        prior: MetalEstimate,
        reference: MetalEstimate | None = None,
    ) -> ClassComparison:
        lutum, om = sample["lutum"], self.om_per_corg * sample["corg"]
        observed = sample_class({m: sample[m] for m in self.metals}, self.limits, self.soil_correction, lutum, om)

        def probabilities(est: MetalEstimate) -> dict[str, float]:
            return sample_class_probabilities(est, self.limits, self.soil_correction, lutum, om,
                                              n_draws=self.n_draws, rng=self._rng)

        return ClassComparison(
            time_utc=t,
            station=station,
            observed=QUALITY_CLASSES[observed],
            probabilities=probabilities(prior),
            reference_probabilities=probabilities(reference) if reference is not None else None,
        )


def score_class_comparisons(comparisons: Sequence[ClassComparison], reference: bool = False) -> dict:
    """`class_scores` over the twin's predictions, or the reference model's if `reference`."""
    rows = [(c.reference_probabilities if reference else c.probabilities, c.observed) for c in comparisons]
    rows = [(p, obs) for p, obs in rows if p is not None]
    if not rows:
        raise ValueError("No class comparisons to score.")
    probs = np.array([[p[k] for k in QUALITY_CLASSES] for p, _ in rows])
    observed = np.array([QUALITY_CLASSES.index(obs) for _, obs in rows])
    return class_scores(probs, observed)
