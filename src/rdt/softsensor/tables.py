"""Build the per-lab-sample tables the soft sensors are fitted and evaluated on.

Lab targets (metal concentrations, sample times, station, period) come from model_table.csv; the
sensor inputs are recomputed from readings.csv with `features_table`, so that
  * training inputs come from cleaned readings (`clean_training_readings`), and
  * test inputs come from readings as delivered, with only the automatic arrival rules -
exactly what the live twin computes.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from rdt.data.cleaning import TRAINING_DROP_QC, apply_arrival_rules, clean_training_readings
from rdt.features.windows import PREFIX_TO_SENSOR, features_table

LAB_COLS = ["sample_time_utc", "station", "period"]


def build_table(
    model_table: pd.DataFrame,
    readings: pd.DataFrame,
    metals: Sequence[str],
    features: Sequence[str],
    training: bool,
) -> pd.DataFrame:
    """One row per lab sample: LAB_COLS + metals + features.

    training=True  -> readings cleaned with training-only rules; lab values with flags in
                      TRAINING_DROP_QC set to NaN.
    training=False -> arrival rules only; lab values used as delivered.
    `readings` must already be restricted to the right time range by the caller.
    """
    sensor_ids = [f"{st}_{k}" for st in model_table["station"].unique() for k in PREFIX_TO_SENSOR.values()]
    r = readings[readings["sensor_id"].isin(sensor_ids)]
    r = clean_training_readings(r) if training else apply_arrival_rules(r)

    frames = []
    for st, g in model_table.sort_values("sample_time_utc").groupby("station"):
        f = features_table(r, st, g["sample_time_utc"], features)
        out = g[LAB_COLS + list(metals)].reset_index(drop=True)
        frames.append(pd.concat([out, f.reset_index(drop=True)], axis=1))
    table = pd.concat(frames, ignore_index=True)

    if training:
        for m in metals:
            lab = readings[readings["sensor_id"].astype(str).str.endswith(f"_spm_{m}")]
            bad = lab[lab["qc"].astype("string").isin(TRAINING_DROP_QC).fillna(False).to_numpy()]
            if len(bad):
                st = bad["sensor_id"].astype(str).str.split("_").str[0]
                key = set(zip(st, bad["time_utc"]))
                hit = [(s, t) in key for s, t in zip(table["station"], table["sample_time_utc"])]
                table.loc[np.array(hit), m] = np.nan
    return table
