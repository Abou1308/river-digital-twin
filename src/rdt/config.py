"""Typed configuration loaded from YAML (see config/default.yaml)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"


@dataclass(frozen=True)
class Periods:
    train_end: pd.Timestamp
    replay_start: pd.Timestamp
    replay_end: pd.Timestamp

    def __post_init__(self) -> None:
        if self.replay_start < self.train_end:
            raise ValueError("replay_start must not be before train_end (data leakage).")
        if self.replay_end <= self.replay_start:
            raise ValueError("replay_end must be after replay_start.")


@dataclass(frozen=True)
class ReplayConfig:
    speed: float | None
    tiers: tuple[str, ...]


@dataclass(frozen=True)
class BrokerConfig:
    kind: str
    mqtt: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TwinConfig:
    estimate_interval_h: float
    history_maxlen: int
    components: tuple[str, ...] = ()   # representation-model components, see runtime.make_components


@dataclass(frozen=True)
class SoftSensorConfig:
    kind: str
    metals: tuple[str, ...]
    features: tuple[str, ...]
    lab_delay_d: float                  # assumed lab turnaround used in the live replay
    eval_lab_delays_d: tuple[float, ...]  # turnarounds compared in offline evaluation
    reference: str | None = None        # model run alongside for the live comparison (None = off)
    recalibrate_prior_n: float | None = None  # online range recalibration (None = off)
    kalman_noise_fit: str = "variogram"  # variogram | one_step, see softsensor/kalman.py


@dataclass(frozen=True)
class MonitoringConfig:
    drift: str                          # cusum | interval_exceedance
    cusum_delta: float                  # sd of bias ignored per result
    cusum_threshold: float              # sd of accumulated bias that raises an alarm


@dataclass(frozen=True)
class QualityConfig:
    om_per_corg: float                  # organic matter / organic carbon, for the soil-type correction


@dataclass(frozen=True)
class DashboardConfig:
    host: str
    port: int
    refresh_ms: int


@dataclass(frozen=True)
class Config:
    data_dir: Path
    periods: Periods
    replay: ReplayConfig
    broker: BrokerConfig
    twin: TwinConfig
    softsensor: SoftSensorConfig
    monitoring: MonitoringConfig
    quality: QualityConfig
    representation: dict[str, dict[str, Any]]   # settings per twin component
    stations: dict[str, dict[str, Any]]
    reaches: list[dict[str, Any]]
    dashboard: DashboardConfig


def _ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value).tz_convert("UTC") if pd.Timestamp(value).tzinfo else pd.Timestamp(value, tz="UTC")


def load_config(path: str | Path | None = None) -> Config:
    """Load a YAML config file (defaults to config/default.yaml)."""
    path = Path(path) if path else DEFAULT_CONFIG
    raw = yaml.safe_load(path.read_text())
    root = path.resolve().parent.parent

    data_dir = Path(raw["paths"]["data_dir"])
    if not data_dir.is_absolute():
        data_dir = root / data_dir

    p = raw["periods"]
    return Config(
        data_dir=data_dir,
        periods=Periods(_ts(p["train_end"]), _ts(p["replay_start"]), _ts(p["replay_end"])),
        replay=ReplayConfig(speed=raw["replay"].get("speed"), tiers=tuple(raw["replay"]["tiers"])),
        broker=BrokerConfig(kind=raw["broker"]["kind"], mqtt=raw["broker"].get("mqtt", {})),
        twin=TwinConfig(**{**raw["twin"], "components": tuple(raw["twin"].get("components", ()))}),
        softsensor=SoftSensorConfig(
            kind=raw["softsensor"]["kind"],
            metals=tuple(raw["softsensor"]["metals"]),
            features=tuple(raw["softsensor"]["features"]),
            lab_delay_d=float(raw["softsensor"]["lab_delay_d"]),
            eval_lab_delays_d=tuple(raw["softsensor"]["eval_lab_delays_d"]),
            reference=raw["softsensor"].get("reference"),
            recalibrate_prior_n=raw["softsensor"].get("recalibrate_prior_n"),
            kalman_noise_fit=raw["softsensor"]["kalman_noise_fit"],
        ),
        monitoring=MonitoringConfig(
            drift=raw["monitoring"]["drift"],
            cusum_delta=float(raw["monitoring"]["cusum"]["delta"]),
            cusum_threshold=float(raw["monitoring"]["cusum"]["threshold"]),
        ),
        quality=QualityConfig(om_per_corg=float(raw["quality"]["om_per_corg"])),
        representation=raw.get("representation") or {},
        stations=raw["stations"],
        reaches=raw.get("reaches", []),
        dashboard=DashboardConfig(**raw["dashboard"]),
    )
