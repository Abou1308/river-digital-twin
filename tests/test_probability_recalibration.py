"""Online Platt recalibration (rdt.monitoring.probability_recalibration)."""

import numpy as np
import pytest

from rdt.monitoring.probability_recalibration import fit_platt, online_platt


def _overconfident(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """True probability q; the forecast doubles its logit (too extreme)."""
    rng = np.random.default_rng(seed)
    q = rng.uniform(0.05, 0.95, n)
    logit_q = np.log(q / (1 - q))
    p = 1 / (1 + np.exp(-2 * logit_q))
    return p, (rng.uniform(size=n) < q).astype(float)


def test_fit_platt_recovers_overconfidence():
    p, o = _overconfident(20000)
    a, b = fit_platt(p, o, np.ones(len(p)))
    assert a == pytest.approx(0, abs=0.05)
    assert b == pytest.approx(0.5, abs=0.05)


def test_calibrated_forecasts_are_left_alone():
    p = np.array([0.2, 0.5, 0.8])
    a, b = fit_platt(p, p, np.ones(3))               # fractional outcomes equal to the forecasts
    assert (a, b) == pytest.approx((0.0, 1.0), abs=1e-6)


def test_online_platt_uses_only_arrived_outcomes():
    p, o = _overconfident(400, seed=1)
    t = np.arange(400.0)
    out = online_platt(p, o, t, available_time=t + 10, prior_n=20)
    assert np.allclose(out[:11], p[:11])             # nothing has arrived yet
    flipped = o.copy()
    flipped[300:] = 1 - flipped[300:]                # changing later outcomes cannot change earlier forecasts
    assert np.allclose(online_platt(p, flipped, t, t + 10, prior_n=20)[:311], out[:311])
    late = slice(200, None)
    assert np.mean((out[late] - o[late]) ** 2) < np.mean((p[late] - o[late]) ** 2)
