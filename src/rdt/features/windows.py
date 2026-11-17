"""Sensor-window inputs for the soft sensor, computed the same way offline (training) and live (twin).

The soft sensor estimates metal levels; turbidity, discharge and SPM concentration are only its
*inputs*. We compute just the few inputs the model uses (config `softsensor.features`), from raw
readings, because
  * training uses cleaned readings, while model_table.csv was built from all readings (flags included);
  * the live twin receives readings over the broker and must compute exactly what training saw.

Definitions (verified against model_table.csv, see tests/test_features.py):
  <sensor>_<stat>_<window>   readings with time in [t - window, t); stat = mean | max;
                             window = 24h | 3d | 7d | 30d
  doy_sin, doy_cos           sin/cos(2*pi*day_of_year/365.25) of t
Sensor prefixes: turb = turbidity, q = discharge, spm = spm_daily (24-h composite SPM).
"""

from __future__ import annotations

import bisect
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Protocol

import numpy as np
import pandas as pd

from rdt.data.cleaning import passes_arrival_rules
from rdt.messaging.messages import SensorMessage

PREFIX_TO_SENSOR = {"turb": "turbidity", "q": "discharge", "spm": "spm_daily"}
_STATS = ("mean", "max")
_WINDOWS = {"24h": pd.Timedelta("24h"), "3d": pd.Timedelta("3D"), "7d": pd.Timedelta("7D"),
            "30d": pd.Timedelta("30D")}
SEASON = ("doy_sin", "doy_cos")


def parse_feature(name: str) -> tuple[str, str, pd.Timedelta]:
    """'q_mean_7d' -> ('discharge', 'mean', 7 days)."""
    prefix, stat, window = name.split("_")
    if prefix not in PREFIX_TO_SENSOR or stat not in _STATS or window not in _WINDOWS:
        raise ValueError(f"Unsupported feature '{name}'")
    return PREFIX_TO_SENSOR[prefix], stat, _WINDOWS[window]


def max_window(names: Iterable[str]) -> pd.Timedelta:
    """Longest sensor window among the features (how much history a warm start needs)."""
    return max((parse_feature(n)[2] for n in names if n not in SEASON), default=pd.Timedelta(0))


def season(t: datetime | pd.Timestamp) -> dict[str, float]:
    a = 2 * math.pi * pd.Timestamp(t).day_of_year / 365.25
    return {"doy_sin": math.sin(a), "doy_cos": math.cos(a)}


def _window_stat(times_ns: Sequence[int], values: Sequence[float], t_ns: int, w_ns: int, stat: str) -> float:
    lo = bisect.bisect_left(times_ns, t_ns - w_ns)
    hi = bisect.bisect_left(times_ns, t_ns)          # excludes readings at exactly t
    if hi <= lo:
        return float("nan")
    v = np.asarray(values[lo:hi], float)
    return float(v.mean() if stat == "mean" else v.max())


def compute_features(
    series: dict[str, tuple[Sequence[int], Sequence[float]]], t: datetime | pd.Timestamp, names: Iterable[str]
) -> dict[str, float]:
    """Features at time t. `series[sensor_kind] = (sorted times in ns, values)` for one station."""
    t_ns = pd.Timestamp(t).value
    out: dict[str, float] = {}
    for name in names:
        if name in SEASON:
            out[name] = season(t)[name]
            continue
        kind, stat, window = parse_feature(name)
        times, values = series.get(kind, ((), ()))
        out[name] = _window_stat(times, values, t_ns, window.value, stat)
    return out


def features_table(readings: pd.DataFrame, station: str, times: Iterable, names: Sequence[str]) -> pd.DataFrame:
    """Offline: features for one station at many times, from long-format readings (already cleaned)."""
    series = {}
    for kind in PREFIX_TO_SENSOR.values():
        r = readings[readings["sensor_id"] == f"{station}_{kind}"].sort_values("time_utc")
        # pandas 3 parses timestamps at microsecond resolution; window bounds are in nanoseconds.
        times_ns = pd.DatetimeIndex(r["time_utc"]).as_unit("ns").asi8.tolist()
        series[kind] = (times_ns, r["value"].to_numpy(float).tolist())
    times = list(times)
    return pd.DataFrame([compute_features(series, t, names) for t in times], columns=list(names),
                        index=pd.DatetimeIndex(times, name="time_utc"))


class FeatureBuilder(Protocol):
    def observe(self, msg: SensorMessage, station: str, kind: str) -> None: ...
    def features(self, station: str, t: datetime) -> dict[str, float]: ...


class LiveFeatureBuilder:
    """Keeps the incoming sensor readings per station and computes features on request.

    Only the automatic arrival rules (`passes_arrival_rules`) are applied: no later QC is available live.
    """

    def __init__(self, names: Sequence[str]) -> None:
        self.names = tuple(names)
        self._series: dict[str, dict[str, tuple[list[int], list[float]]]] = defaultdict(
            lambda: defaultdict(lambda: ([], []))
        )

    def observe(self, msg: SensorMessage, station: str, kind: str) -> None:
        if kind not in PREFIX_TO_SENSOR.values() or not passes_arrival_rules(kind, msg.value):
            return
        times, values = self._series[station][kind]
        t = pd.Timestamp(msg.time_utc).value
        if times and t < times[-1]:                  # out of order: insert in place
            i = bisect.bisect_right(times, t)
            times.insert(i, t)
            values.insert(i, msg.value)
        else:
            times.append(t)
            values.append(msg.value)

    def preload(self, readings: pd.DataFrame) -> None:
        """Warm start from past readings (long format, any order), e.g. the week before the replay starts,
        so that 7-day windows are complete from the first message on. Same rules as `observe`."""
        for sid, t, v in readings.sort_values("time_utc")[["sensor_id", "time_utc", "value"]].itertuples(
                index=False, name=None):
            station, kind = str(sid).split("_", 1)
            self.observe(SensorMessage(str(sid), t.to_pydatetime(), float(v)), station, kind)

    def features(self, station: str, t: datetime) -> dict[str, float]:
        return compute_features(self._series.get(station, {}), t, self.names)
