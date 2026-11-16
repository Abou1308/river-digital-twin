"""Data cleaning: what is removed, and when.

Two levels, following the rule "clean the training data, leave the test data as delivered":

1. Arrival rules (`passes_arrival_rules`): automatic checks a live system can apply the moment a
   reading arrives. Applied identically to training and to the live/test stream, otherwise the
   model would see different inputs live than in training.
     - value must be finite
     - turbidity, discharge and SPM concentration must be > 0 (2,693 Eijsden turbidity values
       of -1.7..0 and 11 negative Eijsden discharges in the dataset are physically impossible)

2. Training-only rules (`clean_training_readings`): use Rijkswaterstaat validation flags that are
   assigned after the fact and are therefore NOT available live. Applied to training data only.
     - qc 50 (implausible) removed
     - qc 20 / 25 (interpolated in time / space and time) removed: not real measurements
   Kept: 00, 03 (larger spread than described), 90 (deviating but approved after validation) and the
   undocumented codes 1001 / 1004 / 1031 (large share of discharge/turbidity; meaning unknown).
"""

from __future__ import annotations

import math

import pandas as pd

POSITIVE_KINDS = ("turbidity", "discharge", "spm_daily")
TRAINING_DROP_QC = frozenset({"50", "20", "25"})


def passes_arrival_rules(kind: str, value: float) -> bool:
    if value is None or not math.isfinite(value):
        return False
    if kind in POSITIVE_KINDS and value <= 0:
        return False
    return True


def _kind(sensor_id: pd.Series) -> pd.Series:
    return sensor_id.astype(str).str.split("_", n=1).str[1]


def apply_arrival_rules(readings: pd.DataFrame) -> pd.DataFrame:
    """Offline version of `passes_arrival_rules` for a long-format readings frame."""
    kind = _kind(readings["sensor_id"])
    v = readings["value"].astype(float)
    ok = v.notna() & ~(kind.isin(POSITIVE_KINDS) & (v <= 0))
    return readings[ok.to_numpy()]


def clean_training_readings(readings: pd.DataFrame) -> pd.DataFrame:
    """Arrival rules + removal of readings with post-hoc validation flags (training data only)."""
    r = apply_arrival_rules(readings)
    return r[~r["qc"].astype("string").isin(TRAINING_DROP_QC).fillna(False).to_numpy()]
