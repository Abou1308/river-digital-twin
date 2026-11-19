"""CUSUM drift detector (rdt.monitoring.cusum)."""

import math
from datetime import datetime, timedelta, timezone

import numpy as np

from rdt.monitoring.cusum import CusumDetector
from rdt.softsensor import MetalEstimate

T0 = datetime(2022, 1, 1, tzinfo=timezone.utc)


def _prior(k: int, station: str = "lobith") -> MetalEstimate:
    """Standard-normal prediction in log space, so an observation exp(z) has innovation z."""
    return MetalEstimate(station, T0 + timedelta(days=7 * k), ("cd", "zn"), np.zeros(2), np.eye(2))


def _run(det: CusumDetector, zs, metal="cd", station="lobith"):
    return [det.observe(_prior(k, station), metal, math.exp(z), False) for k, z in enumerate(zs)]


def test_small_persistent_bias_raises_alarm():
    # z = +1 adds 0.5 per result: the sum passes 5 on the 11th result.
    alerts = _run(CusumDetector(), [1.0] * 11)
    assert alerts[:10] == [None] * 10
    assert alerts[10].kind == "drift" and "above" in alerts[10].message and alerts[10].metal == "cd"


def test_downward_drift_and_reset():
    det = CusumDetector()
    alerts = _run(det, [-1.5] * 6)                  # -1.5 adds 1.0 per result: alarm on the 6th
    assert alerts[5] is not None and "below" in alerts[5].message
    assert _run(det, [-1.5] * 5) == [None] * 5      # sums were reset after the alarm


def test_unbiased_noise_stays_quiet():
    zs = np.random.default_rng(0).normal(0, 1, 200)
    assert sum(a is not None for a in _run(CusumDetector(), zs)) <= 1


def test_state_is_per_station_and_metal():
    det = CusumDetector()
    _run(det, [1.0] * 10, metal="cd")
    assert _run(det, [1.0], metal="zn") == [None]
    assert _run(det, [1.0], station="eijsden") == [None]
    assert _run(det, [1.0], metal="cd")[0] is not None


def test_censored_results_are_ignored():
    det = CusumDetector()
    assert all(det.observe(_prior(k), "cd", math.exp(3.0), True) is None for k in range(20))
