"""Hydrological regime component (rdt.representation.regime)."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from rdt.messaging.messages import SensorMessage
from rdt.representation.regime import HydroRegime, RegimeThresholds, fit_thresholds, next_regime
from rdt.state import InMemoryStateStore

TH = RegimeThresholds(peak=3000.0, high=1800.0, trend=100.0)
T0 = datetime(2022, 1, 1, tzinfo=timezone.utc)


def test_regime_state_machine():
    assert next_regime(None, 1500, None, TH) == "base flow"
    assert next_regime("base flow", 2000, 150, TH) == "rising"
    assert next_regime("rising", 2500, 20, TH) == "rising"          # stays rising while discharge still increases
    assert next_regime("rising", 3100, 50, TH) == "peak"
    assert next_regime("peak", 2900, -10, TH) == "recession"        # falling limb after the peak
    assert next_regime("recession", 1700, -50, TH) == "base flow"   # back below the high threshold
    assert next_regime("base flow", 2000, -150, TH) == "recession"
    assert next_regime("base flow", 2000, 50, TH) == "base flow"    # a weak trend does not start a new state


def test_fit_thresholds_uses_24h_means_and_their_change():
    t = pd.date_range("2015-01-01", periods=24 * 200, freq="1h", tz="UTC")
    q = 1000 + 500 * np.sin(2 * np.pi * np.arange(len(t)) / (24 * 50))       # 50-day cycle
    readings = pd.DataFrame({"sensor_id": "lobith_discharge", "time_utc": t, "value": q})
    th = fit_thresholds(readings, "lobith", peak_pct=90, high_pct=50, trend_pct=75)
    assert th.high == pytest.approx(1000, abs=20)
    assert 1400 < th.peak < 1500
    assert 0 < th.trend < 500 * 2 * np.pi / 50                       # below the steepest daily change
    with pytest.raises(ValueError, match="eijsden"):
        fit_thresholds(readings, "eijsden", 90, 50, 75)


def _hydrograph() -> list[tuple[float, float]]:
    """(hours since T0, discharge): 3 days at 1500, a 2-day rise to 3500, 1 day at peak, a 3-day fall to 1500."""
    q = [1500.0] * 72 + list(np.linspace(1500, 3500, 48)) + [3500.0] * 24 + list(np.linspace(3500, 1500, 72)) \
        + [1500.0] * 72
    return [(h, v) for h, v in enumerate(q)]


def test_component_follows_a_flood_wave(catalog):
    store, regime = InMemoryStateStore(), HydroRegime({"lobith": TH})
    info = catalog["lobith_discharge"]
    seen = []
    for h, q in _hydrograph():
        for minute in (0, 30):
            regime.on_reading(info, SensorMessage("lobith_discharge", T0 + timedelta(hours=h, minutes=minute), q), store)
        state = store.entity("regime", "lobith")
        if state is not None and (not seen or seen[-1] != state.values["regime"]):
            seen.append(state.values["regime"])
    assert seen == ["base flow", "rising", "peak", "recession", "base flow"]
    transitions = [a for a in store.alerts(limit=100) if a.kind == "regime_transition"]
    assert len(transitions) == 4
    assert transitions[0].message.startswith("lobith: recession -> base flow")    # newest first


def test_other_sensors_and_bad_values_are_ignored(catalog):
    store, regime = InMemoryStateStore(), HydroRegime({"lobith": TH})
    for h in range(30):
        regime.on_reading(catalog["lobith_turbidity"], SensorMessage("lobith_turbidity", T0 + timedelta(hours=h), 5.0), store)
        regime.on_reading(catalog["lobith_discharge"], SensorMessage("lobith_discharge", T0 + timedelta(hours=h), -1.0), store)
    assert store.entity("regime", "lobith") is None
