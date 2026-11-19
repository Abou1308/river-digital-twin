"""Sensor health component (rdt.representation.sensor_health)."""

from datetime import datetime, timedelta, timezone

import pandas as pd

from rdt.messaging.messages import SensorMessage
from rdt.representation.sensor_health import SensorHealth, nominal_intervals
from rdt.state import InMemoryStateStore

T0 = datetime(2022, 1, 1, tzinfo=timezone.utc)


def _health(catalog):
    watch = {s: catalog[s] for s in ("lobith_turbidity", "lobith_spm_cd")}
    intervals = {"lobith_turbidity": pd.Timedelta("1h"), "lobith_spm_cd": pd.Timedelta("14D")}
    return SensorHealth(watch, intervals, stale_factor=3, min_stale_h=6)


def _send(health, catalog, store, sid, hours, value=10.0):
    t = T0 + timedelta(hours=hours)
    health.on_reading(catalog[sid], SensorMessage(sid, t, value), store)


def test_nominal_intervals_parse_typical_intervals():
    sensors = pd.DataFrame({"nominal_interval": ["10min", "~7D"]}, index=["a", "b"])
    assert nominal_intervals(sensors) == {"a": pd.Timedelta(minutes=10), "b": pd.Timedelta(days=7)}


def test_gap_longer_than_limit_marks_stale_and_resume_marks_ok(catalog):
    store, health = InMemoryStateStore(), _health(catalog)
    for h in range(0, 5):                                   # hourly turbidity, then silence
        _send(health, catalog, store, "lobith_turbidity", h)
    for h in range(5, 13):                                  # time passes through another sensor's readings
        _send(health, catalog, store, "lobith_spm_cd", h)
    # limit = max(3 x 1 h, 6 h) = 6 h after the last reading at 4 h
    assert store.entity("sensor", "lobith_turbidity").values == {"status": "stale"}
    assert [a.severity for a in store.alerts()] == ["warning"]
    _send(health, catalog, store, "lobith_turbidity", 13)
    assert store.entity("sensor", "lobith_turbidity").values == {"status": "ok"}
    assert [a.kind for a in store.alerts()] == ["data_gap", "data_gap"]
    assert store.entity("sensor", "lobith_spm_cd").values == {"status": "ok"}   # 42-day limit not reached


def test_reading_failing_arrival_rules_is_rejected(catalog):
    store, health = InMemoryStateStore(), _health(catalog)
    _send(health, catalog, store, "lobith_turbidity", 0, value=-1.0)
    assert store.entity("sensor", "lobith_turbidity").values == {"status": "rejected"}
    _send(health, catalog, store, "lobith_turbidity", 1)
    assert store.entity("sensor", "lobith_turbidity").values == {"status": "ok"}
