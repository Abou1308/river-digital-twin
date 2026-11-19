"""Wires all components together from a `Config`. Used by the CLI, notebooks and tests."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta

import pandas as pd

from rdt.config import Config
from rdt.data import (
    load_model_table,
    load_readings,
    load_regulatory_limits,
    load_sensors,
    load_soil_correction,
    sensor_catalog,
)
from rdt.data.schema import SensorInfo, SensorKind
from rdt.features import LiveFeatureBuilder, max_window
from rdt.messaging import Broker, make_broker
from rdt.monitoring import IntervalExceedanceDetector
from rdt.monitoring.classes import ClassEvaluator
from rdt.replay import Replayer, SimClock
from rdt.softsensor import SoftSensor, build_table, make_soft_sensor
from rdt.softsensor.recalibration import RecalibratedSoftSensor
from rdt.state import InMemoryStateStore
from rdt.twin import DigitalTwin, RiverNetwork

log = logging.getLogger(__name__)


@dataclass
class System:
    cfg: Config
    catalog: dict[str, SensorInfo]
    network: RiverNetwork
    limits: pd.DataFrame
    soil_correction: pd.DataFrame
    store: InMemoryStateStore
    broker: Broker
    soft_sensor: SoftSensor
    twin: DigitalTwin
    replayer: Replayer


def build_system(cfg: Config, speed: float | None | str = "config") -> System:
    """Load data, fit the soft sensor on pre-replay data and connect replay -> broker -> twin -> store.

    `speed="config"` uses cfg.replay.speed; pass None to replay as fast as possible.
    """
    sensors = load_sensors(cfg.data_dir)
    catalog = sensor_catalog(sensors)
    network = RiverNetwork.from_config(cfg, catalog)

    sc, cut, start = cfg.softsensor, cfg.periods.train_end, cfg.periods.replay_start
    modelled = [s.id for s in network.modelled_stations()]

    # --- leakage boundary: fit strictly before train_end, replay from replay_start -------------
    table = load_model_table(cfg.data_dir)
    lab = table[(table["sample_time_utc"] < cut) & table["station"].isin(modelled)]
    train = build_table(lab, load_readings(cfg.data_dir, end=cut), sc.metals, sc.features, training=True)
    soft_sensor = make_soft_sensor(sc.kind, sc.metals, sc.features, lab_delay_d=sc.lab_delay_d)
    if sc.recalibrate_prior_n is not None:
        soft_sensor = RecalibratedSoftSensor(soft_sensor, prior_n=sc.recalibrate_prior_n)
    soft_sensor.fit(train, cutoff=cut)
    log.info("Fitted %s soft sensor on %d lab samples", soft_sensor.name, len(train))
    reference = None
    if sc.reference:
        reference = make_soft_sensor(sc.reference, sc.metals, sc.features, lab_delay_d=sc.lab_delay_d)
        reference.fit(train, cutoff=cut)

    # Sensor inputs: warm start with the week before the replay (past data, no leakage), so 7-day
    # windows are complete from the first message.
    features = LiveFeatureBuilder(sc.features)
    input_ids = [sid for sid, info in catalog.items() if info.station in modelled
                 and info.kind in (SensorKind.TURBIDITY, SensorKind.DISCHARGE, SensorKind.SPM_DAILY)]
    warm = max_window(sc.features) + pd.Timedelta(days=1)
    features.preload(load_readings(cfg.data_dir, start - warm, start, input_ids))

    streamed = [sid for sid, info in catalog.items() if info.tier in cfg.replay.tiers]
    readings = load_readings(cfg.data_dir, start, cfg.periods.replay_end, streamed)
    # Lab results are published after the (assumed) lab turnaround; time_utc stays the sample time.
    # Results whose arrival falls after replay_end are not streamed.
    is_lab = readings["sensor_id"].map(lambda s: catalog[s].is_lab).astype(bool)
    readings["available_utc"] = readings["time_utc"] + is_lab.astype(int) * pd.Timedelta(days=sc.lab_delay_d)
    readings = (readings[readings["available_utc"] < cfg.periods.replay_end]
                .sort_values("available_utc", kind="stable").reset_index(drop=True))
    log.info("Loaded %d readings to replay from %d sensors (lab turnaround %.0f d)",
             len(readings), len(streamed), sc.lab_delay_d)

    limits = load_regulatory_limits(cfg.data_dir)
    soil_correction = load_soil_correction(cfg.data_dir)
    store = InMemoryStateStore(history_maxlen=cfg.twin.history_maxlen)
    broker = make_broker(cfg.broker)
    twin = DigitalTwin(
        network=network,
        catalog=catalog,
        soft_sensor=soft_sensor,
        store=store,
        feature_builder=features,
        drift_detector=IntervalExceedanceDetector(),
        estimate_interval=timedelta(hours=cfg.twin.estimate_interval_h),
        reference=reference,
        class_evaluator=ClassEvaluator(sc.metals, limits, soil_correction, cfg.quality.om_per_corg),
    )
    twin.attach(broker)

    clock = SimClock(cfg.periods.replay_start, cfg.replay.speed if speed == "config" else speed)
    replayer = Replayer(readings, catalog, broker, clock)

    return System(
        cfg=cfg,
        catalog=catalog,
        network=network,
        limits=limits,
        soil_correction=soil_correction,
        store=store,
        broker=broker,
        soft_sensor=soft_sensor,
        twin=twin,
        replayer=replayer,
    )
