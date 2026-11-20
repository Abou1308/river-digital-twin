"""Hydrological regime of each modelled station: base flow, rising, peak or recession (ported from the MVP).

Every hour the component takes the trailing 24-h mean discharge q24 (mean of hourly means, at least 12
hours with data) and its change over the previous 24 h, dq24, and moves a small state machine
(`next_regime`). Thresholds are percentiles of the same statistics over the training period, per
station, so "peak" means the same rarity at the Maas and the Rhine:

  peak       q24 >= P`peak_pct` of training q24
  rising     dq24 > P`trend_pct` of training |dq24| (staying rising only needs dq24 > 0)
  recession  q24 >= P`high_pct` and dq24 < -trend (or falling right after rising / peak / recession)
  base flow  otherwise

Entering rising or recession needs a clear trend, staying in it only the sign, so small wiggles do not
flip the state (hysteresis). The regime matters for the reaches: metal-laden sediment is mostly carried
and deposited during high flows.

Each update writes the "regime" entity of the station. Entering or leaving peak is a
"regime_transition" alert; other changes are frequent (at Eijsden about every two days in 2021-2024,
largely weir operation and small showers) and would bury the warnings, so they are only visible in the
entity.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from rdt.data.cleaning import passes_arrival_rules
from rdt.data.schema import SensorInfo, SensorKind
from rdt.messaging.messages import SensorMessage
from rdt.state import Alert, EntityState, StateStore
from rdt.twin.components import TwinComponent

MIN_HOURS = 12             # hours with data needed in a 24-h window
_HOUR = pd.Timedelta(hours=1)
_DAY = pd.Timedelta(hours=24)


@dataclass(frozen=True)
class RegimeThresholds:
    peak: float            # m3/s, q24 at or above which the river is at peak
    high: float            # m3/s, q24 below which a falling river counts as base flow
    trend: float           # m3/s per 24 h, change of q24 that starts rising / recession


def next_regime(state: str | None, q: float, dq24: float | None, th: RegimeThresholds) -> str:
    """Next regime from the trailing 24-h mean discharge q and its change over the previous 24 h."""
    if q >= th.peak:
        return "peak"
    if dq24 is None:
        return state or "base flow"
    if dq24 > th.trend or (state == "rising" and dq24 > 0):
        return "rising"
    if q >= th.high and (dq24 < -th.trend or (state in ("peak", "rising", "recession") and dq24 < 0)):
        return "recession"
    return "base flow"


def fit_thresholds(readings: pd.DataFrame, station: str, peak_pct: float, high_pct: float,
                   trend_pct: float) -> RegimeThresholds:
    """Thresholds from training discharge readings (long format), with the statistics the component uses."""
    q = readings[readings["sensor_id"] == f"{station}_discharge"]
    if q.empty:
        raise ValueError(f"No training discharge readings for {station}; cannot fit regime thresholds.")
    hourly = q.set_index("time_utc")["value"].astype(float).resample("1h").mean()
    q24 = hourly.rolling(24, min_periods=MIN_HOURS).mean()
    change = (q24 - q24.shift(24)).abs().dropna()
    return RegimeThresholds(peak=float(np.nanpercentile(q24, peak_pct)),
                            high=float(np.nanpercentile(q24, high_pct)),
                            trend=float(np.nanpercentile(change, trend_pct)))


def _window_mean(hours: Mapping[pd.Timestamp, list[float]], start: pd.Timestamp, end: pd.Timestamp) -> float | None:
    """Mean of the hourly means in [start, end), if at least MIN_HOURS hours have data."""
    means = [s / n for h, (s, n) in hours.items() if start <= h < end]
    return float(np.mean(means)) if len(means) >= MIN_HOURS else None


class HydroRegime(TwinComponent):
    def __init__(self, thresholds: Mapping[str, RegimeThresholds]) -> None:
        self.thresholds = dict(thresholds)
        self._hours: dict[str, dict[pd.Timestamp, list[float]]] = {st: {} for st in thresholds}  # hour: [sum, n]
        self._current_hour: dict[str, pd.Timestamp] = {}
        self._regime: dict[str, str | None] = {st: None for st in thresholds}

    def on_reading(self, info: SensorInfo, msg: SensorMessage, store: StateStore) -> None:
        st = info.station
        if (info.kind is not SensorKind.DISCHARGE or st not in self.thresholds
                or not passes_arrival_rules(info.kind.value, msg.value)):
            return
        hour = pd.Timestamp(msg.time_utc).floor("h")
        previous = self._current_hour.get(st)
        if previous is not None and hour > previous:
            self._update(st, hour, msg, store)          # all hours before `hour` are complete
        self._current_hour[st] = hour
        hours = self._hours[st]
        acc = hours.setdefault(hour, [0.0, 0])
        acc[0] += msg.value
        acc[1] += 1
        for h in [h for h in hours if h < hour - 2 * _DAY]:
            del hours[h]

    def _update(self, st: str, hour: pd.Timestamp, msg: SensorMessage, store: StateStore) -> None:
        hours = self._hours[st]
        q24 = _window_mean(hours, hour - _DAY, hour)
        if q24 is None:
            return
        before = _window_mean(hours, hour - 2 * _DAY, hour - _DAY)
        dq24 = q24 - before if before is not None else None
        old = self._regime[st]
        new = next_regime(old, q24, dq24, self.thresholds[st])
        self._regime[st] = new
        if old is not None and new != old and "peak" in (old, new):
            store.record_alert(Alert(msg.arrival_utc, st, "regime_transition",
                                     f"{st}: {old} -> {new} (24-h mean discharge {q24:,.0f} m3/s)"))
        store.record_entity(EntityState("regime", st, msg.arrival_utc,
                                        {"regime": new, "q24": q24, "dq24": dq24}))
