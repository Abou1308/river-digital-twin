"""The orchestrator: consumes sensor messages, keeps state, runs the soft sensor and monitoring."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from rdt.data.schema import SensorInfo, SensorKind
from rdt.features.windows import FeatureBuilder
from rdt.messaging.broker import Broker
from rdt.messaging.messages import SensorMessage
from rdt.monitoring.classes import ClassEvaluator
from rdt.monitoring.drift import DriftDetector
from rdt.softsensor.base import NotReadyError, SoftSensor
from rdt.softsensor.estimate import MetalEstimate
from rdt.twin.network import RiverNetwork
from rdt.state import LabComparison, StateStore

log = logging.getLogger(__name__)
_INPUT_KINDS = (SensorKind.TURBIDITY, SensorKind.DISCHARGE, SensorKind.SPM_DAILY)


class DigitalTwin:
    """Subscribes to all sensor topics and keeps the `StateStore` up to date.

    Per message:
      1. record the reading and advance simulated time to the message's arrival time;
      2. if it is a turbidity / discharge / SPM reading: pass it to the feature builder;
      3. if it is a lab metal result at a modelled station (arriving after the lab turnaround, with
         time_utc = sample time): compare it with what the twin predicted *for the sample time using
         only results known at the sample time* (-> drift detector, stored comparison), then let the
         soft sensor assimilate it at the sample time;
      4. if it is a lab lutum / organic-carbon result: pass it to the soft sensor (needed for the
         soil-type correction of the quality class);
      5. once all metals, lutum and organic carbon of a lab sample are in: compare the sample's quality
         class with the class probabilities predicted for the sample time (`class_evaluator`);
      6. every `estimate_interval` of simulated time per station, produce a fresh estimate.
    """

    def __init__(
        self,
        network: RiverNetwork,
        catalog: dict[str, SensorInfo],
        soft_sensor: SoftSensor,
        store: StateStore,
        feature_builder: FeatureBuilder | None = None,
        drift_detector: DriftDetector | None = None,
        estimate_interval: timedelta = timedelta(hours=6),
        reference: SoftSensor | None = None,
        class_evaluator: ClassEvaluator | None = None,
    ) -> None:
        """`reference`: a second soft sensor run alongside (same messages, same lab delays), only so
        the dashboard can show how the twin compares with it live. It does not affect the twin."""
        self.network = network
        self.catalog = catalog
        self.soft_sensor = soft_sensor
        self.store = store
        self.feature_builder = feature_builder
        self.drift_detector = drift_detector
        self.estimate_interval = estimate_interval
        self.reference = reference
        self.class_evaluator = class_evaluator
        self._modelled = {s.id for s in network.modelled_stations()}
        self._last_estimate: dict[str, datetime] = {}

    def attach(self, broker: Broker, pattern: str = "rws/#") -> None:
        broker.subscribe(pattern, self.on_message)

    # ------------------------------------------------------------------------------------------
    def on_message(self, topic: str, msg: SensorMessage) -> None:
        info = self.catalog.get(msg.sensor_id)
        if info is None:
            log.warning("Message for unknown sensor %s on %s", msg.sensor_id, topic)
            return
        now = msg.arrival_utc
        self.store.record_reading(msg)
        self.store.set_sim_time(now)

        if info.station not in self._modelled:
            return
        if info.kind in _INPUT_KINDS and self.feature_builder is not None:
            self.feature_builder.observe(msg, info.station, info.kind.value)
        elif info.kind is SensorKind.SPM_METAL:
            self._handle_lab_metal(info, msg)
            self._collect_class_sample(info.station, msg, info.metal)
        elif info.kind is SensorKind.SPM_PROPERTY:
            self.soft_sensor.assimilate_property(info.station, msg.time_utc, info.prop, msg.value, msg.censored)
            self._collect_class_sample(info.station, msg, info.prop)
        self._maybe_estimate(info.station, now)

    def _handle_lab_metal(self, info: SensorInfo, msg: SensorMessage) -> None:
        station, metal, t = info.station, info.metal, msg.time_utc
        feats = self._features(station, t)
        prior = self._predict(station, t, feats, as_of=t)
        ref = self._predict(station, t, feats, as_of=t, sensor=self.reference) if self.reference else None
        if prior is not None and metal in prior.metals:
            lo, hi = prior.interval(0.9)
            cmp = LabComparison(
                time_utc=msg.time_utc,
                station=station,
                metal=metal,
                observed=msg.value,
                predicted_median=prior.median()[metal],
                lower=lo[metal],
                upper=hi[metal],
                censored=msg.censored,
                reference_median=ref.median()[metal] if ref is not None else None,
            )
            self.store.record_comparison(cmp)
            if self.drift_detector is not None:
                alert = self.drift_detector.observe(prior, metal, msg.value, msg.censored)
                if alert is not None:
                    self.store.record_alert(alert)
        for s in (self.soft_sensor, self.reference):
            if s is not None:
                s.assimilate(station, t, metal, msg.value, msg.censored, features=feats,
                             available_utc=msg.arrival_utc)

    def _collect_class_sample(self, station: str, msg: SensorMessage, key: str) -> None:
        if self.class_evaluator is None:
            return
        sample = self.class_evaluator.observe(station, msg.time_utc, key, msg.value)
        if sample is None:
            return
        t = msg.time_utc
        feats = self._features(station, t)
        prior = self._predict(station, t, feats, as_of=t)
        if prior is None:
            return
        ref = self._predict(station, t, feats, as_of=t, sensor=self.reference) if self.reference else None
        self.store.record_class_comparison(self.class_evaluator.compare(station, t, sample, prior, ref))

    def _maybe_estimate(self, station: str, t: datetime) -> None:
        last = self._last_estimate.get(station)
        if last is not None and t - last < self.estimate_interval:
            return
        feats = self._features(station, t)
        est = self._predict(station, t, feats)
        if est is not None:
            self.store.record_estimate(est)
            self._last_estimate[station] = t
            if self.reference is not None:
                ref = self._predict(station, t, feats, sensor=self.reference)
                if ref is not None:
                    self.store.record_reference_estimate(ref)

    def _features(self, station: str, t: datetime) -> dict[str, float]:
        return self.feature_builder.features(station, t) if self.feature_builder else {}

    def _predict(self, station: str, t: datetime, features: dict[str, float],
                 as_of: datetime | None = None, sensor: SoftSensor | None = None) -> MetalEstimate | None:
        try:
            return (sensor or self.soft_sensor).predict(station, t, features, as_of=as_of)
        except NotReadyError:
            return None
