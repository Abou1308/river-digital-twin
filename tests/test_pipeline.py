"""End-to-end on synthetic data: replay -> broker -> twin -> store."""

from datetime import timedelta

import pandas as pd
import pytest

from rdt.data.schema import METALS
from rdt.messaging import InMemoryBroker
from rdt.monitoring import IntervalExceedanceDetector
from rdt.replay import Replayer, SimClock
from rdt.softsensor import PersistenceSoftSensor
from rdt.state import InMemoryStateStore
from rdt.twin import DigitalTwin, RiverNetwork

START = pd.Timestamp("2021-01-01", tz="UTC")


def _readings() -> pd.DataFrame:
    rows = []
    for h in range(0, 24 * 14):  # two weeks of hourly turbidity at lobith
        rows.append((START + pd.Timedelta(hours=h), "lobith_turbidity", 20.0, False, "00"))
    for d in (3, 10):           # two lab samples
        t = START + pd.Timedelta(days=d, hours=0, minutes=30)
        rows += [(t, f"lobith_spm_{m}", v, False, "00") for m, v in zip(METALS, [1.2, 60, 46, 75, 450])]
    rows.append((START + pd.Timedelta(days=4), "kampen_spm_zn", 400.0, False, "00"))
    df = pd.DataFrame(rows, columns=["time_utc", "sensor_id", "value", "censored", "qc"])
    return df.sort_values("time_utc", kind="stable").reset_index(drop=True)


def test_replay_through_twin(cfg, catalog, train_table):
    network = RiverNetwork.from_config(cfg, catalog)
    sensor = PersistenceSoftSensor(METALS)
    sensor.fit(train_table, START)
    store = InMemoryStateStore()
    broker = InMemoryBroker()
    twin = DigitalTwin(network, catalog, sensor, store, drift_detector=IntervalExceedanceDetector(),
                       estimate_interval=timedelta(hours=6))
    twin.attach(broker)

    readings = _readings()
    n = Replayer(readings, catalog, broker, SimClock(START, speed=None)).run()

    assert n == len(readings)
    assert len(store.readings("lobith_turbidity")) == 24 * 14
    assert store.sim_time() == readings["time_utc"].iloc[-1].to_pydatetime()
    assert len(store.estimates("lobith")) >= 14 * 4 - 1           # one per 6 h
    assert len(store.comparisons("lobith")) == 2 * len(METALS)    # every lab metal compared before update
    assert store.estimates("kampen") == []                         # not a modelled station
    assert store.latest_reading("kampen_spm_zn").value == 400.0
    # after assimilating, persistence predicts the last lab value
    assert store.estimates("lobith")[-1].median()["cd"] == pytest.approx(1.2)


def test_lab_turnaround(cfg, catalog, train_table):
    """Lab results published 5 days after sampling: compared with the prediction for the sample time,
    sim time follows the arrival, and the result is not used before it arrives."""
    sensor = PersistenceSoftSensor(METALS)
    sensor.fit(train_table, START)
    store = InMemoryStateStore()
    broker = InMemoryBroker()
    twin = DigitalTwin(RiverNetwork.from_config(cfg, catalog), catalog, sensor, store,
                       estimate_interval=timedelta(hours=6))
    twin.attach(broker)
    r = _readings()
    lab = r["sensor_id"].str.contains("_spm_")
    r["available_utc"] = r["time_utc"] + lab.astype(int) * pd.Timedelta(days=5)
    r = r.sort_values("available_utc", kind="stable").reset_index(drop=True)
    Replayer(r, catalog, broker, SimClock(START, speed=None)).run()

    cmp = store.comparisons("lobith")
    assert len(cmp) == 2 * len(METALS)
    sample_times = {t.to_pydatetime() for t in r.loc[r["sensor_id"].str.startswith("lobith_spm_"), "time_utc"]}
    assert {c.time_utc for c in cmp} == sample_times
    assert store.sim_time() == r["available_utc"].iloc[-1].to_pydatetime()
    # the second sample (day 10) was compared with a prior that already knew the first (arrived day 8)
    second_cd = [c for c in cmp if c.metal == "cd"][1]
    assert second_cd.predicted_median == pytest.approx(1.2)
    # estimates made before the first result arrived (day 8) still use the training value
    early = [e for e in store.estimates("lobith") if e.time_utc < (START + pd.Timedelta(days=8)).to_pydatetime()]
    assert early and all(e.median()["cd"] != pytest.approx(1.2) for e in early)


def test_reference_scoreboard(cfg, catalog, train_table):
    """The reference model runs alongside; the dashboard scoreboard and running-error chart build."""
    from rdt.dashboard.views import running_error, scoreboard_rows

    twin_sensor, ref = PersistenceSoftSensor(METALS), PersistenceSoftSensor(METALS)
    twin_sensor.fit(train_table, START)
    ref.fit(train_table, START)
    store, broker = InMemoryStateStore(), InMemoryBroker()
    DigitalTwin(RiverNetwork.from_config(cfg, catalog), catalog, twin_sensor, store, reference=ref).attach(broker)
    Replayer(_readings(), catalog, broker, SimClock(START, speed=None)).run()

    cmp = store.comparisons("lobith")
    assert all(c.reference_median == pytest.approx(c.predicted_median) for c in cmp)  # same model
    assert len(store.reference_estimates("lobith")) == len(store.estimates("lobith"))
    rows = scoreboard_rows(cmp, METALS)
    assert [r["verdict"] for r in rows] == ["about equal"] * len(METALS)
    assert len(running_error(cmp, "cd").data) == 2


def test_class_comparison_per_complete_sample(cfg, catalog, train_table):
    """Each lab sample with all metals, lutum and organic carbon gives one class comparison, scored
    for the twin and the reference model; a sample without organic carbon gives none."""
    from rdt.monitoring.classes import ClassEvaluator, score_class_comparisons
    from rdt.quality import QUALITY_CLASSES

    limits = pd.DataFrame(
        [(m.capitalize(), c, lo, hi) for m in METALS
         for c, lo, hi in zip(QUALITY_CLASSES, [float("nan"), 1, 1000, 5000], [1, 1000, 5000, float("nan")])],
        columns=["parameter", "class_en", "lower_exclusive", "upper_inclusive"])
    soil = pd.DataFrame([(m.capitalize(), 1, 0, 0, 2, float("nan"), 2, float("nan"), 25, 10) for m in METALS],
                        columns=["parameter", "A", "B", "C", "lutum_min_pct", "lutum_max_pct", "om_min_pct",
                                 "om_max_pct", "standard_lutum_pct", "standard_om_pct"]).set_index("parameter")
    sensors = [PersistenceSoftSensor(METALS) for _ in range(2)]
    for s in sensors:
        s.fit(train_table, START)
    store = InMemoryStateStore()
    broker = InMemoryBroker()
    twin = DigitalTwin(RiverNetwork.from_config(cfg, catalog), catalog, sensors[0], store,
                       reference=sensors[1], class_evaluator=ClassEvaluator(METALS, limits, soil, 1.724, n_draws=200))
    twin.attach(broker)
    r = _readings()
    t3, t10 = (START + pd.Timedelta(days=d, minutes=30) for d in (3, 10))
    props = [(t3, "lobith_spm_lutum", 20.0, False, "00"), (t3, "lobith_spm_corg", 3.0, False, "00"),
             (t10, "lobith_spm_lutum", 20.0, False, "00")]
    r = pd.concat([r, pd.DataFrame(props, columns=r.columns)]).sort_values("time_utc", kind="stable")
    Replayer(r.reset_index(drop=True), catalog, broker, SimClock(START, speed=None)).run()

    (cmp,) = store.class_comparisons("lobith")
    assert cmp.time_utc == t3.to_pydatetime()
    # every metal and its prediction lies between 1 and 1000 mg/kg: slightly contaminated
    assert cmp.observed == "slightly contaminated"
    assert sum(cmp.probabilities.values()) == pytest.approx(1.0)
    for ref in (False, True):
        s = score_class_comparisons(store.class_comparisons("lobith"), reference=ref)
        assert s["n"] == 1 and s["accuracy"] == 1.0


def test_components_see_all_readings_and_estimates(cfg, catalog, train_table):
    """A component hears every reading, also from stations the soft sensor does not model, and every
    estimate; what it writes to the store is readable as entity state."""
    from rdt.state import EntityState
    from rdt.twin.components import TwinComponent

    class Recorder(TwinComponent):
        def __init__(self):
            self.stations, self.estimates = set(), 0

        def on_reading(self, info, msg, store):
            self.stations.add(info.station)
            store.record_entity(EntityState("station_seen", info.station, msg.arrival_utc, {"last": msg.value}))

        def on_estimate(self, est, store):
            self.estimates += 1

    sensor = PersistenceSoftSensor(METALS)
    sensor.fit(train_table, START)
    store, broker, rec = InMemoryStateStore(), InMemoryBroker(), Recorder()
    DigitalTwin(RiverNetwork.from_config(cfg, catalog), catalog, sensor, store, components=[rec]).attach(broker)
    Replayer(_readings(), catalog, broker, SimClock(START, speed=None)).run()

    assert rec.stations == {"lobith", "kampen"}
    assert rec.estimates == len(store.estimates("lobith"))
    assert store.entity("station_seen", "kampen").values == {"last": 400.0}
    assert set(store.entities("station_seen")) == {"lobith", "kampen"}
