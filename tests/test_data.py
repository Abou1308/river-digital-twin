"""Checks against the real dataset; skipped when data/raw/dataset is not present."""

import pytest

from rdt.data import load_model_table, load_sensors, sensor_catalog
from rdt.data.schema import SensorKind


@pytest.fixture
def data_dir(cfg):
    if not (cfg.data_dir / "sensors.csv").exists():
        pytest.skip("dataset not available")
    return cfg.data_dir


def test_every_sensor_is_classified(data_dir):
    catalog = sensor_catalog(load_sensors(data_dir))
    assert len(catalog) == 40
    metal_sensors = [s for s in catalog.values() if s.kind is SensorKind.SPM_METAL]
    assert {s.station for s in metal_sensors} == {"eijsden", "lobith", "keizersveer", "kampen"}
    assert all(s.is_lab for s in metal_sensors)


def test_model_table_periods_respect_config(data_dir, cfg):
    table = load_model_table(data_dir)
    train = table[table["period"] == "training"]
    assert train["sample_time_utc"].max() < cfg.periods.train_end
    assert table[table["period"] != "training"]["sample_time_utc"].min() >= cfg.periods.replay_start
