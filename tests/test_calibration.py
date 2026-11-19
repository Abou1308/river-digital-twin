"""Calibration checks (rdt.monitoring.calibration)."""

import numpy as np
import pytest

from rdt.monitoring.calibration import (
    block_bootstrap_mean,
    brier_decomposition,
    pit_coverage,
    pit_lognormal,
    reliability_table,
)


def test_reliability_table_bins():
    p = np.array([0.1, 0.1, 0.9, 0.9, 1.0])
    o = np.array([0, 1, 1, 1, 1])
    t = reliability_table(p, o, n_bins=5)
    assert t["n"].tolist() == [2, 0, 0, 0, 3]                 # p = 1.0 goes in the last bin
    assert t.loc[0, "observed"] == 0.5 and t.loc[4, "observed"] == 1.0
    assert t.loc[4, "mean_forecast"] == pytest.approx(2.8 / 3)
    assert t.loc[0, "observed_lo"] < 0.5 < t.loc[0, "observed_hi"]


def test_brier_decomposition_adds_up_when_forecasts_are_constant_per_bin():
    # Two bins with constant forecasts: the decomposition is then exact.
    p = np.array([0.1] * 10 + [0.7] * 10)
    o = np.array([0] * 9 + [1] + [1] * 5 + [0] * 5)
    d = brier_decomposition(p, o)
    assert d["brier"] == pytest.approx(d["reliability"] - d["resolution"] + d["uncertainty"])
    assert d["reliability"] == pytest.approx(0.5 * 0 + 0.5 * 0.2**2)    # bin 1 perfect, bin 2 off by 0.2


def test_pit_of_calibrated_forecasts_is_uniform():
    rng = np.random.default_rng(0)
    median, sd = np.full(20000, 50.0), 0.3
    upper90 = median * np.exp(1.6448536 * sd)
    obs = median * np.exp(rng.normal(0, sd, 20000))
    cov = pit_coverage(pit_lognormal(obs, median, upper90))
    assert cov == pytest.approx({0.5: 0.5, 0.8: 0.8, 0.9: 0.9, 0.95: 0.95}, abs=0.01)
    assert pit_lognormal([50.0], [50.0], [60.0])[0] == pytest.approx(0.5)


def test_block_bootstrap_mean_contains_mean():
    x = np.random.default_rng(1).normal(0.2, 1, 300)
    lo, hi = block_bootstrap_mean(x, block=5)
    assert lo < x.mean() < hi
