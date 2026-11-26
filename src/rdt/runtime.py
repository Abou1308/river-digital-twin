"""Wires all components together from a `Config`. Used by the CLI, notebooks and tests."""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from dataclasses import dataclass, fields
from datetime import timedelta

import pandas as pd

from rdt.config import Config, MonitoringConfig
from rdt.data import (
    load_model_table,
    load_readings,
    load_regulatory_limits,
    load_sensors,
    load_soil_correction,
    sensor_catalog,
)
from rdt.data.cleaning import clean_training_readings
from rdt.data.schema import SensorInfo, SensorKind
from rdt.features import LiveFeatureBuilder, max_window
from rdt.messaging import Broker, make_broker
from rdt.monitoring import DriftDetector, IntervalExceedanceDetector
from rdt.monitoring.classes import ClassEvaluator
from rdt.monitoring.cusum import CusumDetector
from rdt.replay import Replayer, SimClock
from rdt.representation.regime import HydroRegime, fit_thresholds
from rdt.representation.sensor_health import SensorHealth, nominal_intervals
from rdt.representation.downstream import DownstreamCheck, fit_factors
from rdt.representation.reach import ReachSettings, ReachState
from rdt.representation.turbidity_spm import TurbiditySpmCheck, TurbiditySpmRelation, fit_relation
from rdt.softsensor import SoftSensor, build_table, make_soft_sensor
from rdt.softsensor.recalibration import RecalibratedSoftSensor
from rdt.state import InMemoryStateStore
from rdt.twin import DigitalTwin, RiverNetwork
from rdt.twin.components import TwinComponent

log = logging.getLogger(__name__)


@dataclass
class TwinSystem:
    """The twin side: everything except the data source (replay)."""

    cfg: Config
    catalog: dict[str, SensorInfo]
    network: RiverNetwork
    limits: pd.DataFrame
    soil_correction: pd.DataFrame
    store: InMemoryStateStore
    soft_sensor: SoftSensor
    twin: DigitalTwin


@dataclass
class System(TwinSystem):
    """Twin and replay in one process, connected by `broker`."""

    broker: Broker
    replayer: Replayer


def make_drift_detector(cfg: MonitoringConfig) -> DriftDetector:
    if cfg.drift == "cusum":
        return CusumDetector(cfg.cusum_delta, cfg.cusum_threshold)
    if cfg.drift == "interval_exceedance":
        return IntervalExceedanceDetector()
    raise ValueError(f"Unknown drift detector '{cfg.drift}'. Options: cusum, interval_exceedance")


def make_components(cfg: Config, catalog: dict[str, SensorInfo], sensors: pd.DataFrame, streamed: list[str],
                    modelled: list[str], history: pd.DataFrame, network: RiverNetwork, limits: pd.DataFrame,
                    soil_correction: pd.DataFrame) -> list[TwinComponent]:
    """Representation-model components named in `twin.components`, with their `representation` settings.

    `history`: readings before train_end (training rules applied), for components fitted on training data.
    """

    @functools.cache
    def relations() -> dict[str, TurbiditySpmRelation]:
        return {st: fit_relation(history, st) for st in modelled}      # shared by turbidity_spm and reach

    def reach(p: dict) -> ReachState:
        settings = {rid: ReachSettings(float(s["travel_days"]), tuple(s["share_q"]), tuple(s["share"]))
                    for rid, s in p["reaches"].items()}
        return ReachState(network.reaches, settings, cfg.softsensor.metals, relations(), limits, soil_correction,
                          cfg.quality.om_per_corg, residence_days=p["residence_days"], n_draws=p["n_draws"],
                          class_switch_p=p["class_switch_p"])

    def downstream(p: dict) -> DownstreamCheck:
        travel = {rid: float(s["travel_days"]) for rid, s in cfg.representation["reach"]["reaches"].items()}
        factors = {r.id: fit_factors(history, r, travel[r.id], cfg.softsensor.metals, p["window_days"])
                   for r in network.reaches}
        return DownstreamCheck(network.reaches, travel, factors)

    factories: dict[str, Callable[[dict], TwinComponent]] = {
        "sensor_health": lambda p: SensorHealth({s: catalog[s] for s in streamed}, nominal_intervals(sensors), **p),
        "turbidity_spm": lambda p: TurbiditySpmCheck(relations(), **p),
        "regime": lambda p: HydroRegime({st: fit_thresholds(history, st, **p) for st in modelled}),
        "reach": reach,
        "downstream": downstream,
    }
    unknown = [n for n in cfg.twin.components if n not in factories]
    if unknown:
        raise ValueError(f"Unknown twin components {unknown}. Options: {sorted(factories)}")
    return [factories[n](cfg.representation.get(n, {})) for n in cfg.twin.components]


def build_system(cfg: Config, speed: float | None | str = "config") -> System:
    """Load data, fit the soft sensor on pre-replay data and connect replay -> broker -> twin -> store.

    `speed="config"` uses cfg.replay.speed; pass None to replay as fast as possible.
    """
    broker = make_broker(cfg.broker)
    side = build_twin(cfg)
    side.twin.attach(broker)
    replayer = build_replayer(cfg, side.catalog, broker, speed)
    return System(**{f.name: getattr(side, f.name) for f in fields(TwinSystem)}, broker=broker, replayer=replayer)


def streamed_sensors(cfg: Config, catalog: dict[str, SensorInfo]) -> list[str]:
    """Sensors whose readings the replay publishes (the tiers in `replay.tiers`)."""
    return [sid for sid, info in catalog.items() if info.tier in cfg.replay.tiers]


def build_twin(cfg: Config) -> TwinSystem:
    """Fit the soft sensor and components on data before train_end and build the twin (not yet attached
    to a broker). Reads no data from the replay period except the sensor warm-start week before it."""
    sensors = load_sensors(cfg.data_dir)
    catalog = sensor_catalog(sensors)
    network = RiverNetwork.from_config(cfg, catalog)

    sc, cut, start = cfg.softsensor, cfg.periods.train_end, cfg.periods.replay_start
    modelled = [s.id for s in network.modelled_stations()]

    # --- leakage boundary: fit strictly before train_end, replay from replay_start -------------
    table = load_model_table(cfg.data_dir)
    lab = table[(table["sample_time_utc"] < cut) & table["station"].isin(modelled)]
    history = load_readings(cfg.data_dir, end=cut)
    train = build_table(lab, history, sc.metals, sc.features, training=True)
    soft_sensor = make_soft_sensor(sc.kind, sc.metals, sc.features, lab_delay_d=sc.lab_delay_d,
                                   noise_fit=sc.kalman_noise_fit)
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

    limits = load_regulatory_limits(cfg.data_dir)
    soil_correction = load_soil_correction(cfg.data_dir)
    store = InMemoryStateStore(history_maxlen=cfg.twin.history_maxlen)
    twin = DigitalTwin(
        network=network,
        catalog=catalog,
        soft_sensor=soft_sensor,
        store=store,
        feature_builder=features,
        drift_detector=make_drift_detector(cfg.monitoring),
        estimate_interval=timedelta(hours=cfg.twin.estimate_interval_h),
        reference=reference,
        class_evaluator=ClassEvaluator(sc.metals, limits, soil_correction, cfg.quality.om_per_corg),
        components=make_components(cfg, catalog, sensors, streamed_sensors(cfg, catalog), modelled,
                                   clean_training_readings(history), network, limits, soil_correction),
    )
    return TwinSystem(cfg=cfg, catalog=catalog, network=network, limits=limits, soil_correction=soil_correction,
                      store=store, soft_sensor=soft_sensor, twin=twin)


def build_replayer(cfg: Config, catalog: dict[str, SensorInfo], broker: Broker,
                   speed: float | None | str = "config") -> Replayer:
    """Replayer for the replay period, publishing to `broker`; `speed` as in `build_system`."""
    sc = cfg.softsensor
    readings = load_readings(cfg.data_dir, cfg.periods.replay_start, cfg.periods.replay_end,
                             streamed_sensors(cfg, catalog))
    # Lab results are published after the (assumed) lab turnaround; time_utc stays the sample time.
    # Results whose arrival falls after replay_end are not streamed.
    is_lab = readings["sensor_id"].map(lambda s: catalog[s].is_lab).astype(bool)
    readings["available_utc"] = readings["time_utc"] + is_lab.astype(int) * pd.Timedelta(days=sc.lab_delay_d)
    readings = (readings[readings["available_utc"] < cfg.periods.replay_end]
                .sort_values("available_utc", kind="stable").reset_index(drop=True))
    log.info("Loaded %d readings to replay (lab turnaround %.0f d)", len(readings), sc.lab_delay_d)
    clock = SimClock(cfg.periods.replay_start, cfg.replay.speed if speed == "config" else speed)
    return Replayer(readings, catalog, broker, clock)
