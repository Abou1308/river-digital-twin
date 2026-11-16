"""Readers for the CSV files in data/raw/dataset/Full (see DATASET.md there).

All timestamps are returned as timezone-aware UTC. Nothing is resampled or filled here: cleaning
decisions (QC flags, censored values) belong to the consumers, which keep the flags.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd


def load_sensors(data_dir: Path) -> pd.DataFrame:
    """Sensor metadata, indexed by sensor_id."""
    return pd.read_csv(Path(data_dir) / "sensors.csv").set_index("sensor_id")


def load_readings(
    data_dir: Path,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    sensor_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Long-format readings (time_utc, sensor_id, value, censored, qc, rating), sorted by time.

    `start` is inclusive, `end` exclusive.
    """
    df = pd.read_csv(
        Path(data_dir) / "readings.csv",
        dtype={"sensor_id": "category", "qc": "string", "rating": "string"},
    )
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    mask = pd.Series(True, index=df.index)
    if start is not None:
        mask &= df["time_utc"] >= start
    if end is not None:
        mask &= df["time_utc"] < end
    if sensor_ids is not None:
        mask &= df["sensor_id"].isin(list(sensor_ids))
    return df.loc[mask].sort_values("time_utc", kind="stable").reset_index(drop=True)


def load_model_table(data_dir: Path) -> pd.DataFrame:
    """One row per lab sample with metal targets and pre-computed sensor-window features."""
    df = pd.read_csv(Path(data_dir) / "model_table.csv")
    df["sample_time_utc"] = pd.to_datetime(df["sample_time_utc"], utc=True)
    return df


def load_regulatory_limits(data_dir: Path) -> pd.DataFrame:
    """Quality-class limits (Regeling bodemkwaliteit 2022), for a standard soil (25% lutum, 10% OM)."""
    return pd.read_csv(Path(data_dir) / "regulatory_limits.csv")


def load_soil_correction(data_dir: Path) -> pd.DataFrame:
    """Soil-type correction constants A, B, C per substance, indexed by parameter (e.g. 'Cd')."""
    return pd.read_csv(Path(data_dir) / "soil_correction.csv").set_index("parameter")
