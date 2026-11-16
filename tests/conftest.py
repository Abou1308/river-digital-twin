from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rdt.config import DEFAULT_CONFIG, load_config
from rdt.data.schema import METALS, sensor_catalog


@pytest.fixture
def sensors() -> pd.DataFrame:
    rows = []
    for st in ("eijsden", "lobith"):
        rows += [
            (f"{st}_turbidity", st, "periodic", "NTU", f"rws/{st}/turbidity", "core"),
            (f"{st}_discharge", st, "periodic", "m3/s", f"rws/{st}/discharge", "core"),
            (f"{st}_spm_daily", st, "periodic", "mg/l", f"rws/{st}/spm_daily", "core"),
            (f"{st}_spm_lutum", st, "event-based", "% dw", f"rws/{st}/spm/lutum", "core"),
            (f"{st}_spm_corg", st, "event-based", "% dw", f"rws/{st}/spm/corg", "core"),
        ]
        rows += [(f"{st}_spm_{m}", st, "event-based", "mg/kg dw", f"rws/{st}/spm/{m}", "core") for m in METALS]
    rows.append(("kampen_spm_zn", "kampen", "event-based", "mg/kg dw", "rws/kampen/spm/zn", "optional"))
    df = pd.DataFrame(rows, columns=["sensor_id", "station", "type", "unit", "topic", "tier"])
    return df.set_index("sensor_id")


@pytest.fixture
def catalog(sensors):
    return sensor_catalog(sensors)


@pytest.fixture
def train_table() -> pd.DataFrame:
    """Weekly synthetic lab samples for 2019-2020 at both stations."""
    rng = np.random.default_rng(0)
    frames = []
    for st, base in (("eijsden", [6, 80, 55, 110, 700]), ("lobith", [1.4, 65, 47, 80, 470])):
        t = pd.date_range("2019-01-01", "2020-12-28", freq="7D", tz="UTC")
        vals = np.exp(np.log(base) + rng.normal(0, 0.2, (len(t), len(METALS))))
        df = pd.DataFrame(vals, columns=list(METALS))
        df.insert(0, "sample_time_utc", t)
        df.insert(1, "station", st)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def cfg():
    return load_config(DEFAULT_CONFIG)
