"""Fresh deposit of a reach: what the recently settled suspended matter contains (ported from the MVP).

Rijkswaterstaat and Deltares state that the quality of fresh riverbed sediment is set by the suspended
matter that settles. Without transport physics, the deposit is approximated as the sediment-load-weighted
average of the metal content arriving at the reach, with exponential memory:

    weight(t) = sum_i load_i dt_i exp(-(t - t_i) / residence)
    content(t) = sum_i load_i dt_i exp(-(t - t_i) / residence) content_i / weight(t)

so a flood that carries much sediment dominates the deposit for about `residence_days`. No source gives
the residence time; 30 days is the MVP's default and analysis tests other values.

Uncertainty: the content is held as joint draws (n_draws x metals). Every input uses the same standard
normal numbers, so a draw that is high now is high at every time. Prediction errors are thereby treated as
fully correlated in time (the conservative choice, as in the MVP: averaging does not shrink them), and
correlation between metals is kept, which the worst-metal class needs.

Soil properties (lutum, organic carbon) are averaged the same way, from the lab values in force at each
input, because the class limits depend on them: correcting a 30-day deposit with the last single lab
sample would make its class jump with every sample.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


class Deposit:
    def __init__(self, residence_days: float, n_metals: int, n_draws: int = 2000, seed: int = 0) -> None:
        self.residence = pd.Timedelta(days=residence_days)
        self._z = np.random.default_rng(seed).standard_normal((n_draws, n_metals))
        self._mass = np.zeros((n_draws, n_metals))   # sum of weight x content draws
        self._weight = 0.0                           # sum of load x days, decayed (t)
        self._prop_sum: dict[str, float] = {}        # per soil property: sum of weight x value
        self._prop_weight: dict[str, float] = {}     # per soil property: sum of weight of inputs that had it
        self._t: pd.Timestamp | None = None

    @property
    def ready(self) -> bool:
        return self._weight > 0

    def add(self, t: pd.Timestamp, load_t_per_d: float, dt_days: float, mean_log: np.ndarray,
            cov_log: np.ndarray, properties: Mapping[str, float] | None = None) -> None:
        """Add `dt_days` of sediment load (tonnes per day) with log-normal content (mg/kg) at time t, and
        the soil properties known for it (e.g. {"lutum": %, "corg": %})."""
        if not np.isfinite(load_t_per_d) or load_t_per_d < 0:
            raise ValueError(f"Sediment load must be a non-negative number, got {load_t_per_d}.")
        if self._t is not None:
            if t < self._t:
                raise ValueError(f"Deposit input at {t} is before the previous input at {self._t}.")
            decay = float(np.exp(-(t - self._t) / self.residence))
            self._mass *= decay
            self._weight *= decay
            for name in self._prop_sum:
                self._prop_sum[name] *= decay
                self._prop_weight[name] *= decay
        self._t = t
        w = load_t_per_d * dt_days
        if w == 0:
            return
        draws = np.exp(mean_log + self._z @ np.linalg.cholesky(cov_log).T)
        self._mass += w * draws
        self._weight += w
        for name, value in (properties or {}).items():
            self._prop_sum[name] = self._prop_sum.get(name, 0.0) + w * value
            self._prop_weight[name] = self._prop_weight.get(name, 0.0) + w

    def draws(self) -> np.ndarray:
        """Joint draws of the deposit's metal content in mg/kg dw, shape (n_draws, n_metals)."""
        if not self.ready:
            raise ValueError("The deposit has received no sediment yet.")
        return self._mass / self._weight

    def soil_property(self, name: str) -> float | None:
        """Load-weighted average of a soil property over the inputs that had it, or None."""
        weight = self._prop_weight.get(name, 0.0)
        return self._prop_sum[name] / weight if weight > 0 else None
