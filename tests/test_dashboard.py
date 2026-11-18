"""Quality-class panel: summary of the latest estimate and its rendering."""

from datetime import datetime, timezone

import numpy as np
import pytest
from test_quality import LIMITS, SOIL

from rdt.dashboard import create_app
from rdt.dashboard.app import _class_station
from rdt.dashboard.views import class_summary
from rdt.messaging.messages import SensorMessage
from rdt.softsensor import MetalEstimate
from rdt.state import ClassComparison, InMemoryStateStore
from rdt.twin import RiverNetwork

T = datetime(2022, 1, 1, tzinfo=timezone.utc)
EST = MetalEstimate("lobith", T, ("cd", "zn"), np.log([1.0, 600.0]), np.eye(2) * 0.01)
LUTUM = SensorMessage("lobith_spm_lutum", datetime(2021, 12, 1, tzinfo=timezone.utc), 25.0)
CORG = SensorMessage("lobith_spm_corg", datetime(2021, 12, 8, tzinfo=timezone.utc), 10 / 1.724)


def test_class_summary_standard_soil():
    s = class_summary(EST, LUTUM, CORG, LIMITS, SOIL, 1.724)
    # Zn 600 is moderately contaminated in standard soil (25% lutum, 10% organic matter)
    assert s.most_likely == "moderately contaminated"
    assert sum(s.probabilities.values()) == pytest.approx(1.0)
    assert s.om_pct == pytest.approx(10.0)
    assert s.soil_sample_time == LUTUM.time_utc                     # the older of the two
    assert class_summary(EST, LUTUM, CORG, LIMITS, SOIL, 1.724) == s  # same numbers on every refresh


def test_class_summary_without_inputs():
    assert class_summary(None, LUTUM, CORG, LIMITS, SOIL, 1.724).reason == "no estimate yet"
    s = class_summary(EST, LUTUM, None, LIMITS, SOIL, 1.724)
    assert s.probabilities is None and s.most_likely is None
    assert "organic carbon unknown" in s.reason


def test_class_station_renders():
    cmp = [ClassComparison(T, "lobith", "moderately contaminated",
                           {"clean": 0, "slightly contaminated": 0.2, "moderately contaminated": 0.8,
                            "heavily contaminated": 0},
                           {"clean": 0, "slightly contaminated": 1, "moderately contaminated": 0,
                            "heavily contaminated": 0})]
    ok = str(_class_station("lobith", class_summary(EST, LUTUM, CORG, LIMITS, SOIL, 1.724), cmp))
    assert "most likely moderately contaminated" in ok and "twin Brier 0.08" in ok and "last lab value Brier 2.00" in ok
    missing = str(_class_station("lobith", class_summary(EST, None, CORG, LIMITS, SOIL, 1.724), []))
    assert "lutum / organic carbon unknown" in missing and "no complete lab sample yet" in missing


def test_create_app_needs_soil_correction(cfg, catalog):
    network = RiverNetwork.from_config(cfg, catalog)
    with pytest.raises(ValueError, match="soil_correction"):
        create_app(InMemoryStateStore(), network, ("cd", "zn"), LIMITS)
    app = create_app(InMemoryStateStore(), network, ("cd", "zn"), LIMITS, soil_correction=SOIL, om_per_corg=1.724)
    assert "class-probs.children" in app.callback_map
