"""Online recalibration of prediction ranges (rdt.softsensor.recalibration)."""

import math

import numpy as np
import pandas as pd
import pytest

from rdt.data.schema import METALS
from rdt.softsensor import PersistenceSoftSensor
from rdt.softsensor.recalibration import RecalibratedSoftSensor

START = pd.Timestamp("2021-01-01", tz="UTC")
Z90 = 1.6448536269514722


def _wrapped(train_table, prior_n=20):
    inner = PersistenceSoftSensor(METALS)
    s = RecalibratedSoftSensor(inner, prior_n=prior_n)
    s.fit(train_table, START)
    return s


def _assimilate_at_z(s, t, metal, z, delay_d=14):
    """Assimilate a cd result lying z predicted sd above the prediction for its sample time."""
    prior = s.inner.predict("lobith", t, {}, as_of=t)
    i = prior.metals.index(metal)
    value = math.exp(prior.mean_log[i] + z * math.sqrt(prior.cov_log[i, i]))
    s.assimilate("lobith", t, metal, value, False, available_utc=t + pd.Timedelta(days=delay_d))


def test_no_results_leaves_estimate_unchanged(train_table):
    s = _wrapped(train_table)
    a, b = s.predict("lobith", START, {}), s.inner.predict("lobith", START, {})
    assert np.allclose(a.mean_log, b.mean_log) and np.allclose(a.cov_log, b.cov_log)


def test_factor_moves_from_training_towards_observed_errors(train_table):
    s = _wrapped(train_table, prior_n=20)
    for k in range(20):                         # every error exactly twice the 90% half-width
        _assimilate_at_z(s, START + pd.Timedelta(days=7 * k), "cd", 2 * Z90)
    # 20 results against a prior weight of 20: factor^2 = (20 * 1 + 20 * 4) / 40
    assert s.factor("lobith", "cd") == pytest.approx(math.sqrt(2.5))
    assert s.factor("lobith", "zn") == 1.0      # other metals unaffected
    est, inner = s.predict("lobith", START + pd.Timedelta(days=200), {}), s.inner.predict(
        "lobith", START + pd.Timedelta(days=200), {})
    assert np.allclose(est.mean_log, inner.mean_log)           # medians unchanged
    assert est.sd_log()["cd"] == pytest.approx(math.sqrt(2.5) * inner.sd_log()["cd"])


def test_narrows_when_errors_are_small(train_table):
    s = _wrapped(train_table, prior_n=0)
    for k in range(10):
        _assimilate_at_z(s, START + pd.Timedelta(days=7 * k), "zn", 0.5 * Z90)
    assert s.factor("lobith", "zn") == pytest.approx(0.5)


def test_result_not_used_before_it_arrives(train_table):
    s = _wrapped(train_table)
    _assimilate_at_z(s, START, "cd", 4.0, delay_d=14)
    assert s.factor("lobith", "cd", as_of=START + pd.Timedelta(days=10)) == 1.0
    assert s.factor("lobith", "cd", as_of=START + pd.Timedelta(days=15)) > 1.0
