"""The contract every soft sensor implements.

Lifecycle inside the twin:
    fit(training table, cutoff)              once, offline, on lab samples strictly before `cutoff`
    predict(station, t, features, as_of)     estimate at time t from sensor inputs at t, using only lab
                                             results that had arrived strictly before `as_of`
                                             (default: every result assimilated so far)
    assimilate(station, t, metal, value, censored, features, available_utc)
                                             a lab result for a sample taken at t arrives at
                                             `available_utc` (lab turnaround); `features` are the
                                             sensor inputs at the sample time t
    assimilate_property(...)                 latest lutum / organic carbon (soil-type correction)

Training table: one row per lab sample with columns sample_time_utc, station, the metals (mg/kg dw,
NaN if missing) and the feature columns (see `softsensor.tables.build_table`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from rdt.softsensor.estimate import MetalEstimate


class NotReadyError(RuntimeError):
    """Raised by `predict` when the model cannot produce an estimate yet (not fitted / no state)."""


def check_no_leakage(table: pd.DataFrame, cutoff: pd.Timestamp, time_col: str = "sample_time_utc") -> None:
    """Fail loudly if any training row is at or after the replay cutoff."""
    latest = table[time_col].max()
    if pd.notna(latest) and latest >= cutoff:
        raise ValueError(f"Training data reaches {latest}, which is not before the cutoff {cutoff}.")


@dataclass(frozen=True)
class LabObservation:
    sample_time: pd.Timestamp
    available: pd.Timestamp
    metal: str
    value: float
    censored: bool
    features: Mapping[str, float] = field(default_factory=dict)


class FeatureTransform:
    """Sensor inputs -> model matrix: ln of sensor statistics (dilution acts multiplicatively), season
    terms as they are; missing values replaced by the training median; standardised on training."""

    def __init__(self, names: Sequence[str]) -> None:
        self.names = tuple(names)
        self._log = np.array([not n.startswith("doy_") for n in self.names])

    def _raw(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, float, copy=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            X[:, self._log] = np.log(np.where(X[:, self._log] > 0, X[:, self._log], np.nan))
        return X

    def fit(self, df: pd.DataFrame) -> FeatureTransform:
        X = self._raw(df[list(self.names)].to_numpy(float))
        self.median_ = np.nanmedian(X, axis=0)
        Xf = np.where(np.isnan(X), self.median_, X)
        self.mean_, self.sd_ = Xf.mean(0), Xf.std(0)
        self.sd_[self.sd_ == 0] = 1.0
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X = self._raw(df[list(self.names)].to_numpy(float))
        X = np.where(np.isnan(X), self.median_, X)
        return (X - self.mean_) / self.sd_

    def transform_one(self, features: Mapping[str, float] | None) -> np.ndarray:
        features = features or {}
        row = pd.DataFrame([[features.get(n, np.nan) for n in self.names]], columns=list(self.names))
        return self.transform(row)[0]


class SoftSensor(ABC):
    name: str = "base"

    def __init__(self, metals: Sequence[str], features: Sequence[str] = ()) -> None:
        self.metals = tuple(metals)
        self.features = tuple(features)
        # Latest lab lutum / organic carbon per station: {station: {"lutum": (t, value), "corg": ...}}
        self._sediment_props: dict[str, dict[str, tuple[datetime, float]]] = {}
        self._obs: dict[str, list[LabObservation]] = {}

    @abstractmethod
    def fit(self, train: pd.DataFrame, cutoff: pd.Timestamp) -> None:
        """Fit on training-table rows. Implementations must call `check_no_leakage(train, cutoff)`."""

    @abstractmethod
    def predict(
        self, station: str, t: datetime, features: Mapping[str, float], as_of: datetime | None = None
    ) -> MetalEstimate:
        """Estimate all metals at `station` at time `t`."""

    def assimilate(
        self,
        station: str,
        t: datetime,
        metal: str,
        value: float,
        censored: bool,
        features: Mapping[str, float] | None = None,
        available_utc: datetime | None = None,
    ) -> None:
        """Store a lab result; subclasses update their state in `_on_observation`."""
        if metal not in self.metals or not np.isfinite(value) or value <= 0:
            return
        obs = LabObservation(pd.Timestamp(t), pd.Timestamp(available_utc if available_utc is not None else t),
                             metal, float(value), bool(censored), dict(features or {}))
        self._obs.setdefault(station, []).append(obs)
        self._on_observation(station, obs)

    def _on_observation(self, station: str, obs: LabObservation) -> None:  # noqa: B027 - optional hook
        pass

    def known(self, station: str, as_of: datetime | None) -> list[LabObservation]:
        """Assimilated results that had arrived strictly before `as_of` (all if None)."""
        obs = self._obs.get(station, [])
        if as_of is None:
            return list(obs)
        cut = pd.Timestamp(as_of)
        return [o for o in obs if o.available < cut]

    def assimilate_property(self, station: str, t: datetime, prop: str, value: float, censored: bool) -> None:
        """Store the latest lutum / organic carbon (needed for the soil-type correction)."""
        self._sediment_props.setdefault(station, {})[prop] = (t, value)

    def sediment_properties(self, station: str) -> dict[str, tuple[datetime, float]]:
        return dict(self._sediment_props.get(station, {}))
