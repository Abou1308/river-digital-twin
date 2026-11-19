"""Turbidity - suspended matter relation and a consistency check on it (ported from the MVP).

Fitted per station on training readings:  ln(SPM) = a + b ln(1 + turbidity_24h),  where turbidity_24h is
the mean turbidity in the 24 h before a daily SPM composite (at least 12 readings), SPM is uncensored.

Used by
  * `TurbiditySpmCheck` (component "turbidity_spm"): each new daily composite gives the standardised
    residual z; a CUSUM on z flags the two sensors drifting apart (turbidity sensor fouling or
    recalibration, or a change in the kind of sediment);
  * the reach flux component: SPM from turbidity on days without a daily composite.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from rdt.data.cleaning import passes_arrival_rules
from rdt.data.schema import SensorInfo, SensorKind
from rdt.messaging.messages import SensorMessage
from rdt.monitoring.cusum import Cusum
from rdt.state import Alert, EntityState, StateStore
from rdt.twin.components import TwinComponent

MIN_READINGS = 12          # turbidity readings needed in the 24 h before a composite (hourly sensor)
_DAY = pd.Timedelta(hours=24)


@dataclass(frozen=True)
class TurbiditySpmRelation:
    a: float
    b: float
    sd: float              # residual sd of ln(SPM)
    n: int                 # daily composites used in the fit

    def log_spm(self, turbidity_24h: float) -> float:
        return self.a + self.b * math.log1p(max(turbidity_24h, 0.0))

    def z(self, spm: float, turbidity_24h: float) -> float:
        return (math.log(spm) - self.log_spm(turbidity_24h)) / self.sd


def daily_pairs(readings: pd.DataFrame, station: str) -> pd.DataFrame:
    """Uncensored daily SPM composites with the mean turbidity of the preceding 24 h (long-format readings)."""
    spm = readings[(readings["sensor_id"] == f"{station}_spm_daily") & ~readings["censored"].astype(bool)]
    tu = readings[readings["sensor_id"] == f"{station}_turbidity"].sort_values("time_utc")
    times, vals = pd.DatetimeIndex(tu["time_utc"]), tu["value"].to_numpy(float)
    rows = []
    for t, v in zip(spm["time_utc"], spm["value"]):
        i0, i1 = times.searchsorted(t - _DAY), times.searchsorted(t)
        if i1 - i0 >= MIN_READINGS:
            rows.append((t, float(v), float(vals[i0:i1].mean())))
    return pd.DataFrame(rows, columns=["time_utc", "spm", "turb_24h"])


def fit_relation(readings: pd.DataFrame, station: str) -> TurbiditySpmRelation:
    d = daily_pairs(readings, station)
    if len(d) < 30:
        raise ValueError(f"Only {len(d)} turbidity / SPM pairs at {station}; cannot fit the relation.")
    A = np.column_stack([np.ones(len(d)), np.log1p(np.maximum(d["turb_24h"], 0))])
    y = np.log(d["spm"].to_numpy())
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    return TurbiditySpmRelation(float(coef[0]), float(coef[1]), float(resid.std(ddof=2)), len(d))


class TurbiditySpmCheck(TwinComponent):
    def __init__(self, relations: Mapping[str, TurbiditySpmRelation], delta: float = 0.5,
                 threshold: float = 5.0) -> None:
        self.relations = dict(relations)
        self._cusum = {st: Cusum(delta, threshold) for st in relations}
        self._turbidity: dict[str, deque[tuple[pd.Timestamp, float]]] = {st: deque() for st in relations}

    def on_reading(self, info: SensorInfo, msg: SensorMessage, store: StateStore) -> None:
        st = info.station
        if st not in self.relations or not passes_arrival_rules(info.kind.value, msg.value):
            return
        t = pd.Timestamp(msg.time_utc)
        if info.kind is SensorKind.TURBIDITY:
            window = self._turbidity[st]
            window.append((t, msg.value))
            while window and window[0][0] < t - 2 * _DAY:
                window.popleft()
        elif info.kind is SensorKind.SPM_DAILY and not msg.censored:
            values = [v for ti, v in self._turbidity[st] if t - _DAY <= ti < t]
            if len(values) < MIN_READINGS:
                return
            z = self.relations[st].z(msg.value, float(np.mean(values)))
            store.record_entity(EntityState("turbidity_spm", st, msg.arrival_utc, {"z": z}))
            direction = self._cusum[st].update(z)
            if direction is not None:
                more = "higher" if direction == "up" else "lower"
                store.record_alert(Alert(msg.arrival_utc, st, "sensor_drift",
                                         f"{st}: turbidity and suspended matter drifting apart "
                                         f"(SPM persistently {more} than turbidity implies, CUSUM {direction})",
                                         severity="warning"))
