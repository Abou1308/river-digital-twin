"""Validation: our sensor-input definitions reproduce model_table.csv (built independently).

The metal estimates are the output; these inputs only matter in so far as they are computed the same
way in training and live. model_table.csv was built from all readings (QC flags included), so the
comparison uses uncleaned readings.
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from rdt.config import DEFAULT_CONFIG, load_config
from rdt.features import LiveFeatureBuilder, features_table
from rdt.features.windows import parse_feature, season
from rdt.messaging import SensorMessage

CHECK = ["q_mean_7d", "q_mean_24h", "q_max_7d", "turb_mean_7d", "turb_mean_24h", "turb_max_7d",
         "spm_mean_7d", "doy_sin", "doy_cos"]


@pytest.fixture(scope="module")
def real():
    cfg = load_config(DEFAULT_CONFIG)
    if not (cfg.data_dir / "readings.csv").exists():
        pytest.skip("dataset not available")
    from rdt.data import load_model_table, load_readings
    mt = load_model_table(cfg.data_dir)
    return mt, load_readings(cfg.data_dir, end=pd.Timestamp("2016-01-01", tz="UTC"))


@pytest.mark.parametrize("station", ["eijsden", "lobith"])
def test_features_match_model_table(real, station):
    mt, readings = real
    lab = mt[(mt.station == station) & (mt.sample_time_utc < pd.Timestamp("2016-01-01", tz="UTC"))]
    cols = [c for c in CHECK if c in mt.columns]
    ours = features_table(readings, station, lab.sample_time_utc, cols)
    for c in cols:
        a, b = ours[c].to_numpy(), lab[c].to_numpy(float)
        both = np.isfinite(a) & np.isfinite(b)
        assert both.mean() > 0.9, c
        assert np.allclose(a[both], b[both], rtol=1e-6, atol=1e-9), c


def _msg(h: float, v: float) -> SensorMessage:
    return SensorMessage("lobith_discharge", datetime(2021, 1, 1, tzinfo=timezone.utc) + pd.Timedelta(hours=h), v)


def test_live_builder_window_and_rules():
    fb = LiveFeatureBuilder(["q_mean_24h", "q_max_24h"])
    for h, v in [(0, 1000.0), (1, -5.0), (3, float("nan")), (2, 3000.0), (30, 2000.0)]:  # out of order, 2 rejected
        fb.observe(_msg(h, v), "lobith", "discharge")
    t = datetime(2021, 1, 2, 6, tzinfo=timezone.utc)             # window [00:00 + 6h, 30:00) -> hours 6..29: none
    assert np.isnan(fb.features("lobith", t)["q_mean_24h"])
    t = datetime(2021, 1, 1, 23, tzinfo=timezone.utc)            # hours 0 and 2
    f = fb.features("lobith", t)
    assert f["q_mean_24h"] == pytest.approx(2000.0) and f["q_max_24h"] == 3000.0
    t = datetime(2021, 1, 1, 2, tzinfo=timezone.utc)             # reading at exactly t is excluded
    assert fb.features("lobith", t)["q_mean_24h"] == pytest.approx(1000.0)


def test_parse_and_season():
    assert parse_feature("turb_max_7d") == ("turbidity", "max", pd.Timedelta("7D"))
    with pytest.raises(ValueError):
        parse_feature("q_median_7d")
    s = season(pd.Timestamp("2021-04-01", tz="UTC"))
    assert s["doy_sin"] ** 2 + s["doy_cos"] ** 2 == pytest.approx(1.0)


def test_longer_windows_and_max_window():
    from rdt.features import max_window
    assert parse_feature("q_mean_30d") == ("discharge", "mean", pd.Timedelta("30D"))
    assert parse_feature("turb_mean_3d") == ("turbidity", "mean", pd.Timedelta("3D"))
    assert max_window(["q_mean_7d", "doy_sin", "spm_mean_30d"]) == pd.Timedelta("30D")
