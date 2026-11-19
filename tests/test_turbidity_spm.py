"""Turbidity - suspended matter relation and consistency check (rdt.representation.turbidity_spm)."""

import math

import numpy as np
import pandas as pd
import pytest

from rdt.messaging.messages import SensorMessage
from rdt.representation.turbidity_spm import TurbiditySpmCheck, TurbiditySpmRelation, fit_relation
from rdt.state import InMemoryStateStore

T0 = pd.Timestamp("2019-01-01", tz="UTC")


def _readings(days: int, a=1.0, b=0.9, noise=0.1, seed=0) -> pd.DataFrame:
    """Hourly turbidity and a daily composite at midnight that follows ln SPM = a + b ln(1 + turb)."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(days):
        turb = float(np.exp(rng.normal(2.5, 0.6)))
        for h in range(24):
            rows.append((T0 + pd.Timedelta(days=d, hours=h), "lobith_turbidity", turb, False))
        spm = math.exp(a + b * math.log1p(turb) + rng.normal(0, noise))
        rows.append((T0 + pd.Timedelta(days=d + 1), "lobith_spm_daily", spm, False))
    return pd.DataFrame(rows, columns=["time_utc", "sensor_id", "value", "censored"])


def test_fit_recovers_relation():
    rel = fit_relation(_readings(200), "lobith")
    assert (rel.a, rel.b, rel.sd) == pytest.approx((1.0, 0.9, 0.1), abs=0.05)
    assert rel.n == 200


def test_autocorrelation_scales_cusum_input():
    rel = TurbiditySpmRelation(0.0, 1.0, 0.5, 100, rho=0.6)
    turb = math.e - 1                                  # ln(1 + turb) = 1, so ln SPM is predicted to be 1
    assert rel.z(math.exp(2.0), turb) == pytest.approx(2.0)
    assert rel.z_for_cusum(math.exp(2.0), turb) == pytest.approx(2.0 * math.sqrt(0.4 / 1.6))


def test_fit_needs_enough_pairs():
    with pytest.raises(ValueError, match="cannot fit"):
        fit_relation(_readings(10), "lobith")


def _replay(check, catalog, readings, store, spm_factor=1.0):
    for r in readings.sort_values("time_utc", kind="stable").itertuples():
        value = r.value * (spm_factor if r.sensor_id.endswith("spm_daily") else 1.0)
        check.on_reading(catalog[r.sensor_id], SensorMessage(r.sensor_id, r.time_utc.to_pydatetime(), value), store)


def test_check_quiet_when_consistent_and_alarms_on_persistent_offset(catalog):
    rel = TurbiditySpmRelation(1.0, 0.9, 0.1, 200)
    store = InMemoryStateStore()
    _replay(TurbiditySpmCheck({"lobith": rel}), catalog, _readings(60, seed=1), store)
    assert [a for a in store.alerts() if a.kind == "sensor_drift"] == []
    assert abs(store.entity("turbidity_spm", "lobith").values["z"]) < 4

    store = InMemoryStateStore()                 # SPM 20% higher than turbidity implies: z ~ +1.8 per day
    _replay(TurbiditySpmCheck({"lobith": rel}), catalog, _readings(60, seed=1), store, spm_factor=1.2)
    alerts = [a for a in store.alerts(limit=100) if a.kind == "sensor_drift"]
    assert len(alerts) == 1 and "higher" in alerts[0].message      # one episode, not one alert per alarm
    assert store.entity("turbidity_spm", "lobith").values["status"] == "drifting up"


def test_episode_ends_after_quiet_period(catalog):
    rel = TurbiditySpmRelation(1.0, 0.9, 0.1, 200)
    store, check = InMemoryStateStore(), TurbiditySpmCheck({"lobith": rel}, quiet_days=10)
    shifted, normal = _readings(20, seed=2), _readings(40, seed=3)
    normal["time_utc"] += pd.Timedelta(days=20)
    _replay(check, catalog, shifted, store, spm_factor=1.2)
    _replay(check, catalog, normal, store)
    assert store.entity("turbidity_spm", "lobith").values["status"] == "consistent"
    assert [a.severity for a in store.alerts(limit=100) if a.kind == "sensor_drift"] == ["info", "warning"]
