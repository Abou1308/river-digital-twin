from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from rdt.data.schema import METALS
from rdt.softsensor import MetalEstimate, NotReadyError, PersistenceSoftSensor

CUTOFF = pd.Timestamp("2021-01-01", tz="UTC")
T = datetime(2021, 1, 5, tzinfo=timezone.utc)


def test_estimate_interval_brackets_median():
    est = MetalEstimate("lobith", T, ("cd", "zn"), np.log([1.5, 450.0]), np.diag([0.04, 0.01]))
    lo, hi = est.interval(0.9)
    med = est.median()
    for m in ("cd", "zn"):
        assert lo[m] < med[m] < hi[m]
    assert med["cd"] == pytest.approx(1.5)


def test_fit_rejects_leakage(train_table):
    late = train_table.copy()
    late.loc[0, "sample_time_utc"] = CUTOFF
    with pytest.raises(ValueError, match="cutoff"):
        PersistenceSoftSensor(METALS).fit(late, CUTOFF)


def test_persistence_predict_and_assimilate(train_table):
    s = PersistenceSoftSensor(METALS)
    with pytest.raises(NotReadyError):
        s.predict("lobith", T, {})
    s.fit(train_table, CUTOFF)
    est = s.predict("lobith", T, {})
    assert est.cov_log.shape == (5, 5)
    s.assimilate("lobith", T, "cd", 2.5, censored=False)
    assert s.predict("lobith", T, {}).median()["cd"] == pytest.approx(2.5)
