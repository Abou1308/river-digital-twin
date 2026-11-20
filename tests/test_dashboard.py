"""Quality-class panel: summary of the latest estimate and its rendering."""

from datetime import datetime, timezone

import numpy as np
import pytest
from test_quality import LIMITS, SOIL

from rdt.dashboard import create_app
from rdt.dashboard.app import _class_station, _clicked_reach, _reach_panel
from rdt.dashboard.views import CLASS_COLOURS, class_summary, map_hover, network_map, reach_label
from rdt.messaging.messages import SensorMessage
from rdt.softsensor import MetalEstimate
from rdt.state import ClassComparison, EntityState, InMemoryStateStore
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
    assert "..network-map.figure...reach-details.children.." in app.callback_map


REACH_VALUES = {
    "spm_source": "daily composite", "share": 0.16, "discharge_m3s": 320.0, "spm_mg_l": 40.0,
    "sediment_load_t_per_d": 1105.9, "metal_flux_kg_per_d": {"cd": 1.1, "zn": 663.6},
    "deposit_median": {"cd": 1.0, "zn": 600.0}, "deposit_p05": {"cd": 0.8, "zn": 500.0},
    "deposit_p95": {"cd": 1.3, "zn": 700.0}, "deposit_lutum_pct": 25.0, "deposit_corg_pct": 5.8,
    "class_probabilities": {"clean": 0.0, "slightly contaminated": 0.3, "moderately contaminated": 0.7,
                            "heavily contaminated": 0.0},
    "most_likely_class": "moderately contaminated", "class": "moderately contaminated",
}


def test_network_map_colours_reaches_and_selects_by_reach(cfg, catalog):
    network = RiverNetwork.from_config(cfg, catalog)
    first = network.reaches[0]
    fig = network_map(network, {first.upstream: "clean"}, {first.id: "moderately contaminated"}, {})
    reach_trace, stations = fig.data[0], fig.data[-1]
    assert reach_trace.line.color == CLASS_COLOURS["moderately contaminated"]
    assert fig.data[1].line.color == CLASS_COLOURS[None]                    # second reach: no class yet
    by_station = dict(zip(stations.text, stations.customdata))
    assert by_station[first.upstream.capitalize()] == first.id             # clicking a station selects its reach
    assert by_station[first.downstream.capitalize()] == first.id


def test_map_hover_and_reach_panel(cfg, catalog):
    network = RiverNetwork.from_config(cfg, catalog)
    reach = next(r for r in network.reaches if r.upstream == "lobith")
    regime = EntityState("regime", "lobith", T, {"regime": "rising", "q24": 2500.0, "dq24": 150.0})
    state = EntityState("reach", reach.id, T, REACH_VALUES)
    hover = map_hover(network, {"lobith": "clean"}, {"lobith": regime}, {reach.id: state})
    assert "regime: rising" in hover["lobith"] and "suspended sediment: clean" in hover["lobith"]
    assert "fresh deposit: moderately contaminated" in hover[reach.id] and "1,106 t/day" in hover[reach.id]
    panel = str(_reach_panel(reach_label(network, reach), state, regime))
    assert "Upstream regime: rising" in panel and "16% of the upstream discharge" in panel
    assert "663.6" in panel and "500 – 700" in panel and "Fresh deposit: moderately contaminated" in panel
    early = str(_reach_panel("x", EntityState("reach", reach.id, T, {"spm_source": "none"}), None))
    assert "No flux at this step" in early and "no sediment has arrived yet" in early
    assert "No data for this reach yet" in str(_reach_panel("x", None, None))


def test_clicked_reach():
    assert _clicked_reach(None) is None
    assert _clicked_reach({"points": [{"customdata": "rijn_lobith_kampen"}]}) == "rijn_lobith_kampen"
