"""Live state of the twin, shared between the twin (writer) and the dashboard (reader).

`StateStore` is the boundary between the two. The in-memory implementation works when everything
runs in one process; a persistent implementation (e.g. SQLite / a time-series DB) can replace it
once the twin and dashboard run as separate processes over MQTT.
"""

from __future__ import annotations

import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from rdt.messaging.messages import SensorMessage
from rdt.softsensor.estimate import MetalEstimate


@dataclass(frozen=True)
class Alert:
    time_utc: datetime
    station: str
    kind: str              # e.g. "drift", "class_change", "data_gap"
    message: str
    metal: str | None = None
    severity: str = "info"  # info | warning | critical


@dataclass(frozen=True)
class LabComparison:
    """A lab result next to what the soft sensor predicted for that moment (before assimilating it)."""

    time_utc: datetime
    station: str
    metal: str
    observed: float
    predicted_median: float
    lower: float
    upper: float
    censored: bool
    reference_median: float | None = None   # what the reference model (last lab value) predicted


class StateStore(Protocol):
    def record_reading(self, msg: SensorMessage) -> None: ...
    def record_estimate(self, est: MetalEstimate) -> None: ...
    def record_reference_estimate(self, est: MetalEstimate) -> None: ...
    def record_comparison(self, cmp: LabComparison) -> None: ...
    def record_alert(self, alert: Alert) -> None: ...
    def set_sim_time(self, t: datetime) -> None: ...

    def sim_time(self) -> datetime | None: ...
    def readings(self, sensor_id: str, since: datetime | None = None) -> list[SensorMessage]: ...
    def latest_reading(self, sensor_id: str) -> SensorMessage | None: ...
    def estimates(self, station: str) -> list[MetalEstimate]: ...
    def reference_estimates(self, station: str) -> list[MetalEstimate]: ...
    def comparisons(self, station: str) -> list[LabComparison]: ...
    def alerts(self, limit: int = 50) -> list[Alert]: ...


class InMemoryStateStore:
    """Thread-safe, bounded in-memory store."""

    def __init__(self, history_maxlen: int = 100_000) -> None:
        self._lock = threading.Lock()
        self._readings: dict[str, deque[SensorMessage]] = defaultdict(lambda: deque(maxlen=history_maxlen))
        self._estimates: dict[str, deque[MetalEstimate]] = defaultdict(lambda: deque(maxlen=history_maxlen))
        self._ref_estimates: dict[str, deque[MetalEstimate]] = defaultdict(lambda: deque(maxlen=history_maxlen))
        self._comparisons: dict[str, list[LabComparison]] = defaultdict(list)
        self._alerts: deque[Alert] = deque(maxlen=1000)
        self._sim_time: datetime | None = None

    # -- writes ---------------------------------------------------------------------------
    def record_reading(self, msg: SensorMessage) -> None:
        with self._lock:
            self._readings[msg.sensor_id].append(msg)

    def record_estimate(self, est: MetalEstimate) -> None:
        with self._lock:
            self._estimates[est.station].append(est)

    def record_reference_estimate(self, est: MetalEstimate) -> None:
        with self._lock:
            self._ref_estimates[est.station].append(est)

    def record_comparison(self, cmp: LabComparison) -> None:
        with self._lock:
            self._comparisons[cmp.station].append(cmp)

    def record_alert(self, alert: Alert) -> None:
        with self._lock:
            self._alerts.append(alert)

    def set_sim_time(self, t: datetime) -> None:
        with self._lock:
            self._sim_time = t

    # -- reads (return copies so readers never see a deque being mutated) -------------------
    def sim_time(self) -> datetime | None:
        with self._lock:
            return self._sim_time

    def readings(self, sensor_id: str, since: datetime | None = None) -> list[SensorMessage]:
        """All readings, or only those with time_utc >= since (read from the newest end, so cheap for
        a recent window even with a long history)."""
        with self._lock:
            q = self._readings.get(sensor_id, ())
            if since is None:
                return list(q)
            out = []
            for m in reversed(q):
                if m.time_utc < since:
                    break
                out.append(m)
        return out[::-1]

    def latest_reading(self, sensor_id: str) -> SensorMessage | None:
        with self._lock:
            q = self._readings.get(sensor_id)
            return q[-1] if q else None

    def estimates(self, station: str) -> list[MetalEstimate]:
        with self._lock:
            return list(self._estimates.get(station, ()))

    def reference_estimates(self, station: str) -> list[MetalEstimate]:
        with self._lock:
            return list(self._ref_estimates.get(station, ()))

    def comparisons(self, station: str) -> list[LabComparison]:
        with self._lock:
            return list(self._comparisons.get(station, ()))

    def alerts(self, limit: int = 50) -> list[Alert]:
        with self._lock:
            return list(self._alerts)[-limit:][::-1]
