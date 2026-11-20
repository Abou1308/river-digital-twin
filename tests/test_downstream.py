"""Downstream check (rdt.representation.downstream)."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from rdt.data.schema import METALS
from rdt.messaging.messages import SensorMessage
from rdt.representation.downstream import DownstreamCheck, DownstreamFactor, fit_factors
from rdt.softsensor.estimate import MetalEstimate
from rdt.state import InMemoryStateStore
from rdt.twin.network import Reach

IJSSEL = Reach("rijn_lobith_kampen", "Rijn", "lobith", "kampen")
T0 = datetime(2022, 1, 1, tzinfo=timezone.utc)


def test_fit_factors_takes_median_log_ratio_after_travel_time():
    rng = np.random.default_rng(0)
    up_t = pd.date_range("2012-01-01", "2020-12-01", freq="14D", tz="UTC")
    up = pd.DataFrame({"sensor_id": "lobith_spm_zn", "time_utc": up_t,
                       "value": 400 * np.exp(rng.normal(0, 0.2, len(up_t))), "censored": False})
    # downstream every ~91 days, 2 days after an upstream sample, with 20% lower content
    picks = up.iloc[::6]
    down = pd.DataFrame({"sensor_id": "kampen_spm_zn", "time_utc": picks["time_utc"] + pd.Timedelta(days=2),
                         "value": picks["value"].to_numpy() * 0.8, "censored": False})
    f = fit_factors(pd.concat([up, down]), IJSSEL, travel_days=2, metals=("zn",))["zn"]
    assert f.log_ratio == pytest.approx(np.log(0.8)) and f.sd == pytest.approx(0, abs=1e-9)
    assert f.n == len(down)
    with pytest.raises(ValueError, match="cd"):
        fit_factors(pd.concat([up, down]), IJSSEL, travel_days=2, metals=("cd",))


def test_downstream_result_is_compared_with_upstream_estimate(catalog):
    check = DownstreamCheck([IJSSEL], {IJSSEL.id: 2}, {IJSSEL.id: {"zn": DownstreamFactor(np.log(0.8), 0.1, 30)}})
    store = InMemoryStateStore()
    for h in range(0, 24 * 10, 6):
        mean = np.log([1, 50, 40, 60, 400 if h < 24 * 5 else 800])
        check.on_estimate(MetalEstimate("lobith", T0 + timedelta(hours=h), METALS, mean, np.eye(5) * 0.01), store)
    sample = T0 + timedelta(days=6)                      # left Lobith on day 4, when zn was 400
    msg = SensorMessage("kampen_spm_zn", sample, 330.0, available_utc=sample + timedelta(days=14))
    check.on_reading(catalog["kampen_spm_zn"], msg, store)
    (cmp,) = store.comparisons("kampen")
    assert cmp.predicted_median == pytest.approx(320.0)
    assert cmp.lower < 330 < cmp.upper and not store.alerts()
    check.on_reading(catalog["kampen_spm_zn"], SensorMessage("kampen_spm_zn", sample, 900.0), store)
    assert store.alerts()[0].kind == "downstream_mismatch"


def test_no_comparison_without_an_upstream_estimate_near_the_departure_time(catalog):
    check = DownstreamCheck([IJSSEL], {IJSSEL.id: 2}, {IJSSEL.id: {"zn": DownstreamFactor(0.0, 0.1, 30)}})
    store = InMemoryStateStore()
    check.on_estimate(MetalEstimate("lobith", T0, METALS, np.log([1, 50, 40, 60, 400]), np.eye(5) * 0.01), store)
    check.on_reading(catalog["kampen_spm_zn"], SensorMessage("kampen_spm_zn", T0 + timedelta(days=9), 400.0), store)
    assert store.comparisons("kampen") == []
