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
    r["available_utc"] = r["time_utc"] + lab * pd.Timedelta(days=5)
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
