"""Sensor health: is each sensor delivering usable readings?

Entity "sensor" per sensor id, with status
  ok        the last reading passed the arrival rules (rdt/data/cleaning.py);
  rejected  the last reading failed them (e.g. a negative turbidity);
  stale     no reading for more than `stale_factor` x the sensor's nominal interval (sensors.csv), and
            never less than `min_stale_h` hours, so short gaps in 10-minute series do not count.
A sensor going stale raises a "data_gap" warning; its next reading raises a "data_gap" info alert.
Staleness is checked once per simulated hour. Times are arrival times (lab results arrive after the
lab turnaround).
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import timedelta

import pandas as pd

from rdt.data.cleaning import passes_arrival_rules
from rdt.data.schema import SensorInfo
from rdt.messaging.messages import SensorMessage
from rdt.state import Alert, EntityState, StateStore
from rdt.twin.components import TwinComponent

_CHECK_EVERY = pd.Timedelta(hours=1)


def nominal_intervals(sensors: pd.DataFrame) -> dict[str, pd.Timedelta]:
    """{sensor_id: nominal interval} from sensors.csv; '~7D' (typical, event-based) is read as 7 days."""
    return {str(sid): pd.Timedelta(str(v).lstrip("~")) for sid, v in sensors["nominal_interval"].items()}


class SensorHealth(TwinComponent):
    def __init__(self, sensors: Mapping[str, SensorInfo], intervals: Mapping[str, pd.Timedelta],
                 stale_factor: float = 3.0, min_stale_h: float = 6.0) -> None:
        """`sensors`: the sensors to watch (those that are streamed)."""
        self.sensors = dict(sensors)
        self.limits = {sid: max(stale_factor * intervals[sid], pd.Timedelta(hours=min_stale_h)) for sid in sensors}
        self._last: dict[str, pd.Timestamp] = {}
        self._status: dict[str, str] = {}
        self._start: pd.Timestamp | None = None
        self._last_check: pd.Timestamp | None = None

    def on_reading(self, info: SensorInfo, msg: SensorMessage, store: StateStore) -> None:
        now = pd.Timestamp(msg.arrival_utc)
        if self._start is None:
            self._start = self._last_check = now
        sid = info.sensor_id
        if sid in self.sensors:
            if self._status.get(sid) == "stale":
                gap = now - self._last.get(sid, self._start)
                self._alert(store, now, sid, "info", f"{sid} delivering again after {_days(gap)}")
            self._last[sid] = now
            self._set(store, now, sid, "ok" if passes_arrival_rules(info.kind.value, msg.value) else "rejected")
        if now - self._last_check >= _CHECK_EVERY:
            self._check(store, now)
            self._last_check = now

    def _check(self, store: StateStore, now: pd.Timestamp) -> None:
        for sid, limit in self.limits.items():
            last = self._last.get(sid, self._start)
            if self._status.get(sid) != "stale" and now - last > limit:
                self._set(store, now, sid, "stale")
                self._alert(store, now, sid, "warning", f"{sid}: no reading for {_days(now - last)} "
                                                        f"(stale after {_days(limit)})")

    def _set(self, store: StateStore, now: pd.Timestamp, sid: str, status: str) -> None:
        if self._status.get(sid) == status:
            return
        self._status[sid] = status
        store.record_entity(EntityState("sensor", sid, now.to_pydatetime(), {"status": status}))

    def _alert(self, store: StateStore, now: pd.Timestamp, sid: str, severity: str, message: str) -> None:
        store.record_alert(Alert(now.to_pydatetime(), self.sensors[sid].station, "data_gap", message,
                                 severity=severity))


def _days(gap: timedelta) -> str:
    hours = gap.total_seconds() / 3600
    return f"{hours:.0f} h" if hours < 48 else f"{hours / 24:.0f} days"
