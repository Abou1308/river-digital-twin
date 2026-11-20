"""Reach component (rdt.representation.reach)."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from rdt.data import load_regulatory_limits, load_soil_correction
from rdt.data.schema import METALS
from rdt.messaging.messages import SensorMessage
from rdt.representation.reach import ReachSettings, ReachState, sediment_load
from rdt.representation.turbidity_spm import TurbiditySpmRelation
from rdt.softsensor.estimate import MetalEstimate
from rdt.state import InMemoryStateStore
from rdt.twin.network import Reach

T0 = datetime(2022, 1, 1, tzinfo=timezone.utc)
RHINE = Reach("rijn_lobith_kampen", "Rijn", "lobith", "kampen")
IJSSEL = ReachSettings(travel_days=2, share_q=(680, 1300, 2480, 16000), share=(0.16, 0.18, 0.15, 0.154))
RELATION = TurbiditySpmRelation(a=0.5, b=0.8, sd=0.4, n=100)
CLEAN = np.log([0.3, 20, 20, 30, 100])          # cd, cu, ni, pb, zn: clean at standard soil
DIRTY = np.log([8.0, 300, 100, 600, 2500])      # well into moderately / heavily contaminated


def _component(cfg) -> ReachState:
    return ReachState([RHINE], {RHINE.id: IJSSEL}, METALS, {"lobith": RELATION},
                      load_regulatory_limits(cfg.data_dir), load_soil_correction(cfg.data_dir), 1.97,
                      residence_days=30, n_draws=500)


def _readings(store, hours: range, spm: bool = True):
    for h in hours:
        t = T0 + timedelta(hours=h)
        store.record_reading(SensorMessage("lobith_discharge", t, 2000.0))
        store.record_reading(SensorMessage("lobith_turbidity", t, 30.0))
        if spm and h % 24 == 0:
            store.record_reading(SensorMessage("lobith_spm_daily", t, 50.0))


def _estimate(hours: int, mean_log) -> MetalEstimate:
    return MetalEstimate("lobith", T0 + timedelta(hours=hours), METALS, np.asarray(mean_log), np.eye(5) * 0.01)


def test_share_is_interpolated_on_discharge():
    assert IJSSEL.fraction(2000) == pytest.approx(0.18 - 0.03 * 700 / 1180)
    assert IJSSEL.fraction(100) == 0.16 and IJSSEL.fraction(30000) == 0.154
    assert sediment_load(100.0, 50.0) == pytest.approx(432.0)              # 100 m3/s x 50 g/m3 x 86400 s


def test_flux_then_deposit_after_travel_time(cfg):
    store, reach = InMemoryStateStore(), _component(cfg)
    store.record_reading(SensorMessage("lobith_spm_lutum", T0, 25.0))
    store.record_reading(SensorMessage("lobith_spm_corg", T0, 10 / 1.97))
    _readings(store, range(0, 24 * 6))
    reach.on_estimate(_estimate(24, CLEAN), store)
    assert "sediment_load_t_per_d" not in store.entity("reach", RHINE.id).values   # no interval yet
    reach.on_estimate(_estimate(30, CLEAN), store)
    v = store.entity("reach", RHINE.id).values
    load = sediment_load(IJSSEL.fraction(2000) * 2000, 50.0)
    assert v["sediment_load_t_per_d"] == pytest.approx(load)
    assert v["metal_flux_kg_per_d"]["zn"] == pytest.approx(load * 100 * 1e-3)
    assert v["spm_source"] == "daily composite" and "deposit_median" not in v       # still travelling
    for h in range(36, 24 * 4 + 1, 6):
        reach.on_estimate(_estimate(h, CLEAN), store)
    v = store.entity("reach", RHINE.id).values
    assert v["deposit_median"]["zn"] == pytest.approx(100, rel=0.05)
    assert v["most_likely_class"] == "clean"


def test_turbidity_fallback_without_composite(cfg):
    store, reach = InMemoryStateStore(), _component(cfg)
    _readings(store, range(0, 48), spm=False)
    reach.on_estimate(_estimate(24, CLEAN), store)
    reach.on_estimate(_estimate(30, CLEAN), store)
    v = store.entity("reach", RHINE.id).values
    assert v["spm_source"] == "turbidity"
    assert v["spm_mg_l"] == pytest.approx(np.exp(RELATION.log_spm(30.0)))


def test_class_change_alert_when_deposit_gets_worse(cfg):
    store, reach = InMemoryStateStore(), _component(cfg)
    store.record_reading(SensorMessage("lobith_spm_lutum", T0, 25.0))
    store.record_reading(SensorMessage("lobith_spm_corg", T0, 10 / 1.97))
    _readings(store, range(0, 24 * 40))
    for h in range(24, 24 * 40, 6):
        reach.on_estimate(_estimate(h, CLEAN if h < 24 * 10 else DIRTY), store)
    alerts = [a for a in store.alerts() if a.kind == "class_change"]
    assert alerts and alerts[-1].message.startswith(f"{RHINE.id}: fresh deposit most likely")
    assert all(a.severity == "warning" for a in alerts) and alerts[0].station == "kampen"
    assert store.entity("reach", RHINE.id).values["most_likely_class"] == "heavily contaminated"
