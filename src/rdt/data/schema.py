"""Shared vocabulary: metals, sensor kinds and per-sensor metadata.

Sensor ids follow the pattern ``<station>_<kind>`` (see sensors.csv), e.g. ``lobith_turbidity``,
``eijsden_spm_cd``, ``lobith_water_zn_total``. `SensorInfo` turns that into structured fields so the
rest of the code never has to parse ids by hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

METALS: tuple[str, ...] = ("cd", "cu", "ni", "pb", "zn")
SEDIMENT_PROPERTIES: tuple[str, ...] = ("lutum", "corg")
MODELLED_STATIONS: tuple[str, ...] = ("eijsden", "lobith")


class SensorKind(str, Enum):
    TURBIDITY = "turbidity"           # hourly, NTU/FTU
    DISCHARGE = "discharge"           # 10-min, m3/s
    SPM_DAILY = "spm_daily"           # 24-h composite SPM concentration, mg/l
    SPM_METAL = "spm_metal"           # lab: metal in suspended sediment, mg/kg dw  (soft-sensor target)
    SPM_PROPERTY = "spm_property"     # lab: lutum / organic carbon in SPM, % dw
    WATER = "water"                   # lab: metals / SPM in the water column (optional tier)


@dataclass(frozen=True)
class SensorInfo:
    sensor_id: str
    station: str
    kind: SensorKind
    unit: str
    topic: str
    is_lab: bool                     # event-based lab analysis vs periodic automatic sensor
    tier: str
    metal: str | None = None         # set for SPM_METAL
    prop: str | None = None          # set for SPM_PROPERTY ("lutum" | "corg")


def _classify(sensor_id: str, station: str) -> tuple[SensorKind, str | None, str | None]:
    suffix = sensor_id[len(station) + 1 :]
    if suffix in ("turbidity", "discharge", "spm_daily"):
        return SensorKind(suffix), None, None
    if suffix.startswith("spm_"):
        name = suffix[4:]
        if name in METALS:
            return SensorKind.SPM_METAL, name, None
        if name in SEDIMENT_PROPERTIES:
            return SensorKind.SPM_PROPERTY, None, name
    if suffix.startswith("water_"):
        return SensorKind.WATER, None, None
    raise ValueError(f"Unrecognised sensor id: {sensor_id}")


def sensor_catalog(sensors: pd.DataFrame) -> dict[str, SensorInfo]:
    """Build {sensor_id: SensorInfo} from the sensors table (as returned by `load_sensors`)."""
    catalog: dict[str, SensorInfo] = {}
    for sid, row in sensors.iterrows():
        kind, metal, prop = _classify(str(sid), row["station"])
        catalog[str(sid)] = SensorInfo(
            sensor_id=str(sid),
            station=row["station"],
            kind=kind,
            unit=row["unit"],
            topic=row["topic"],
            is_lab=row["type"] == "event-based",
            tier=row["tier"],
            metal=metal,
            prop=prop,
        )
    return catalog
