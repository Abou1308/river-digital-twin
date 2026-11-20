"""Soft sensors: estimate metal content of suspended sediment between lab samples."""

from collections.abc import Sequence
from typing import Any

from rdt.softsensor.base import LabObservation, NotReadyError, SoftSensor, check_no_leakage
from rdt.softsensor.baselines import PersistenceSoftSensor, RidgeLogSoftSensor
from rdt.softsensor.estimate import MetalEstimate
from rdt.softsensor.kalman import KalmanSoftSensor
from rdt.softsensor.tables import build_table

_REGISTRY: dict[str, type[SoftSensor]] = {
    "persistence": PersistenceSoftSensor,
    "ridge_log": RidgeLogSoftSensor,
    "kalman": KalmanSoftSensor,
}


def make_soft_sensor(kind: str, metals: Sequence[str], features: Sequence[str] = (), **params: Any) -> SoftSensor:
    """`params` are model-specific (e.g. lab_delay_d for ridge_log); unknown ones are ignored."""
    try:
        cls = _REGISTRY[kind]
    except KeyError:
        raise ValueError(f"Unknown soft sensor '{kind}'. Options: {sorted(_REGISTRY)}") from None
    if cls is RidgeLogSoftSensor:
        return cls(metals, features, lab_delay_d=params.get("lab_delay_d", 0.0))
    if cls is KalmanSoftSensor:
        return cls(metals, features, noise_fit=params.get("noise_fit", "variogram"))
    return cls(metals, features)


__all__ = [
    "KalmanSoftSensor",
    "LabObservation",
    "MetalEstimate",
    "NotReadyError",
    "PersistenceSoftSensor",
    "RidgeLogSoftSensor",
    "SoftSensor",
    "build_table",
    "check_no_leakage",
    "make_soft_sensor",
]
