"""Fresh-deposit store (rdt.representation.deposit)."""

import numpy as np
import pandas as pd
import pytest

from rdt.representation.deposit import Deposit

T0 = pd.Timestamp("2022-01-01", tz="UTC")
COV = np.diag([0.01, 0.01])


def _median(dep: Deposit) -> np.ndarray:
    return np.median(dep.draws(), axis=0)


def test_load_weighted_average_with_memory():
    dep = Deposit(residence_days=30, n_metals=2, n_draws=4000)
    dep.add(T0, 100.0, 1.0, np.log([1.0, 100.0]), COV)
    dep.add(T0, 300.0, 1.0, np.log([3.0, 300.0]), COV)                # same time: plain load-weighted mean
    assert _median(dep) == pytest.approx([2.5, 250.0], rel=0.02)
    dep.add(T0 + pd.Timedelta(days=300), 1.0, 1.0, np.log([10.0, 1000.0]), COV)   # old input has decayed away
    assert _median(dep) == pytest.approx([10.0, 1000.0], rel=0.02)


def test_errors_are_fully_correlated_in_time():
    """Averaging two inputs with the same spread keeps that spread (no 1/sqrt(n) shrinking)."""
    dep = Deposit(residence_days=30, n_metals=2, n_draws=20000)
    for day in range(2):
        dep.add(T0 + pd.Timedelta(days=day), 100.0, 1.0, np.log([1.0, 100.0]), np.diag([0.04, 0.04]))
    assert np.std(np.log(dep.draws()[:, 0])) == pytest.approx(0.2, rel=0.05)


def test_not_ready_and_bad_input():
    dep = Deposit(residence_days=30, n_metals=2)
    with pytest.raises(ValueError, match="no sediment"):
        dep.draws()
    dep.add(T0, 0.0, 1.0, np.log([1.0, 100.0]), COV)
    assert not dep.ready
    with pytest.raises(ValueError, match="non-negative"):
        dep.add(T0, float("nan"), 1.0, np.log([1.0, 100.0]), COV)
    with pytest.raises(ValueError, match="before"):
        dep.add(T0 - pd.Timedelta(days=1), 1.0, 1.0, np.log([1.0, 100.0]), COV)
