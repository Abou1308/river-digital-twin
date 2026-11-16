"""Reference models that the Kalman soft sensor has to beat.

PersistenceSoftSensor   prediction = last lab value known at the time
RidgeLogSoftSensor      ridge regression of ln(metal) on sensor inputs + ln(last known lab value) + its
                        age; the best simple model in analysis/03_model_comparison.py
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rdt.softsensor.base import FeatureTransform, NotReadyError, SoftSensor, check_no_leakage
from rdt.softsensor.estimate import MetalEstimate


def _psd(c: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    c = np.nan_to_num((c + c.T) / 2)
    w, v = np.linalg.eigh(c)
    return (v * np.maximum(w, floor)) @ v.T


class _LastKnownMixin(SoftSensor):
    """Last known lab value per metal: from assimilated results, else from the end of training."""

    def _fit_last(self, train: pd.DataFrame) -> None:
        self._train_last: dict[str, dict[str, tuple[pd.Timestamp, float]]] = {}
        for st, g in train.sort_values("sample_time_utc").groupby("station"):
            self._train_last[st] = {}
            for m in self.metals:
                d = g.dropna(subset=[m])
                d = d[d[m] > 0]
                if len(d):
                    self._train_last[st][m] = (d["sample_time_utc"].iloc[-1], float(np.log(d[m].iloc[-1])))

    def _last_known(self, station: str, as_of: datetime | None) -> dict[str, tuple[pd.Timestamp, float]]:
        out = dict(self._train_last.get(station, {}))
        for o in sorted(self.known(station, as_of), key=lambda o: o.sample_time):
            out[o.metal] = (o.sample_time, float(np.log(o.value)))
        return out


class PersistenceSoftSensor(_LastKnownMixin):
    """Prediction = last known lab value; uncertainty = spread of sample-to-sample log changes in training.

    Ignores sensor inputs. Uncertainty does not grow with time since the last sample.
    """

    name = "persistence"

    def fit(self, train: pd.DataFrame, cutoff: pd.Timestamp) -> None:
        check_no_leakage(train, cutoff)
        self._fit_last(train)
        self._cov: dict[str, np.ndarray] = {}
        for st, g in train.sort_values("sample_time_utc").groupby("station"):
            diffs = np.log(g[list(self.metals)].astype(float)).diff()
            self._cov[st] = _psd(diffs.cov().to_numpy())

    def predict(self, station: str, t: datetime, features: Mapping[str, float],
                as_of: datetime | None = None) -> MetalEstimate:
        if station not in getattr(self, "_cov", {}):
            raise NotReadyError(f"No state for station {station}")
        last = self._last_known(station, as_of)
        mean = np.array([last[m][1] if m in last else np.nan for m in self.metals])
        return MetalEstimate(station, t, self.metals, mean, self._cov[station], model=self.name)


class RidgeLogSoftSensor(_LastKnownMixin):
    """Per station and metal: ridge of ln(metal) on transformed sensor inputs, ln(last known lab value)
    and that value's age in days. The lab turnaround `lab_delay_d` must be set, because it decides
    which lab value counts as "last known" in training."""

    name = "ridge_log"

    def __init__(self, metals: Sequence[str], features: Sequence[str] = (), lab_delay_d: float = 0.0) -> None:
        super().__init__(metals, features)
        self.lab_delay_d = float(lab_delay_d)

    def _with_last_known(self, g: pd.DataFrame, m: str) -> pd.DataFrame:
        lab = g[["sample_time_utc", m]].dropna()
        lab = lab[lab[m] > 0].rename(columns={"sample_time_utc": "lab_time", m: "known"})
        lab["available"] = lab["lab_time"] + pd.Timedelta(days=self.lab_delay_d)
        out = pd.merge_asof(g.sort_values("sample_time_utc"), lab.sort_values("available"),
                            left_on="sample_time_utc", right_on="available", allow_exact_matches=False)
        out["prev_log"] = np.log(out["known"])
        out["prev_age_d"] = (out["sample_time_utc"] - out["lab_time"]).dt.total_seconds() / 86400
        return out

    def fit(self, train: pd.DataFrame, cutoff: pd.Timestamp) -> None:
        check_no_leakage(train, cutoff)
        self._fit_last(train)
        self._tf: dict[str, FeatureTransform] = {}
        self._models: dict[str, dict[str, object]] = {}
        self._cov: dict[str, np.ndarray] = {}
        for st, g in train.sort_values("sample_time_utc").groupby("station"):
            tf = FeatureTransform(self.features).fit(g)
            self._tf[st], self._models[st] = tf, {}
            resid = pd.DataFrame(index=g["sample_time_utc"].values, columns=list(self.metals), dtype=float)
            for m in self.metals:
                d = self._with_last_known(g, m).dropna(subset=[m, "prev_log"])
                d = d[d[m] > 0]
                X = np.column_stack([tf.transform(d), d["prev_log"], d["prev_age_d"]])
                y = np.log(d[m].to_numpy(float))
                model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))).fit(X, y)
                self._models[st][m] = model
                resid.loc[d["sample_time_utc"].values, m] = y - model.predict(X)
            self._cov[st] = _psd(resid.cov().to_numpy())

    def predict(self, station: str, t: datetime, features: Mapping[str, float],
                as_of: datetime | None = None) -> MetalEstimate:
        if station not in getattr(self, "_models", {}):
            raise NotReadyError(f"No state for station {station}")
        x = self._tf[station].transform_one(features)
        last = self._last_known(station, as_of)
        tt = pd.Timestamp(t)
        mean = np.full(len(self.metals), np.nan)
        for i, m in enumerate(self.metals):
            if m in last and m in self._models[station]:
                lt, lv = last[m]
                row = np.concatenate([x, [lv, (tt - lt).total_seconds() / 86400]])
                mean[i] = self._models[station][m].predict(row[None, :])[0]
        return MetalEstimate(station, t, self.metals, mean, self._cov[station], model=self.name)
