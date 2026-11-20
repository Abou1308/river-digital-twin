"""Kalman soft sensor on synthetic data with a known structure."""

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from rdt.softsensor import KalmanSoftSensor, NotReadyError

METALS = ("cd", "zn")
FEATS = ("q_mean_7d",)
CUT = pd.Timestamp("2021-01-01", tz="UTC")


@pytest.fixture(scope="module")
def table():
    """ln metal = slowly drifting level - 0.5 * ln(discharge) + noise; 6 years of 2-weekly samples."""
    rng = np.random.default_rng(1)
    t = pd.date_range("2015-01-01", "2020-12-20", freq="14D", tz="UTC")
    n = len(t)
    lnq = np.log(1500) + 0.5 * rng.standard_normal(n)
    level = np.cumsum(rng.normal(0, 0.03, (n, 2)), axis=0) + np.log([1.5, 450])
    y = level - 0.5 * lnq[:, None] + rng.normal(0, 0.1, (n, 2))
    df = pd.DataFrame(np.exp(y), columns=list(METALS))
    df.insert(0, "sample_time_utc", t)
    df.insert(1, "station", "lobith")
    df["q_mean_7d"] = np.exp(lnq)
    return df


@pytest.fixture
def fitted(table):
    k = KalmanSoftSensor(METALS, FEATS)
    k.fit(table, CUT)
    return k


def test_not_ready_and_leakage(table):
    k = KalmanSoftSensor(METALS, FEATS)
    with pytest.raises(NotReadyError):
        k.predict("lobith", CUT, {"q_mean_7d": 1500})
    late = table.copy()
    late.loc[len(late) - 1, "sample_time_utc"] = CUT
    with pytest.raises(ValueError, match="cutoff"):
        k.fit(late, CUT)


def test_learns_dilution_effect(fitted):
    est_lo = fitted.predict("lobith", CUT, {"q_mean_7d": 800})
    est_hi = fitted.predict("lobith", CUT, {"q_mean_7d": 3000})
    # true effect: -0.5 * ln(3000/800)
    d = est_hi.mean_log - est_lo.mean_log
    assert np.all(d < 0)
    assert d == pytest.approx(-0.5 * np.log(3000 / 800), abs=0.2)


def test_uncertainty_grows_and_assimilation_pulls(fitted):
    f = {"q_mean_7d": 1500}
    t1 = CUT + timedelta(days=30)
    sd_near = fitted.predict("lobith", CUT, f).sd_log()["cd"]
    sd_far = fitted.predict("lobith", CUT + timedelta(days=300), f).sd_log()["cd"]
    assert sd_far > sd_near
    before = fitted.predict("lobith", t1, f).median()["cd"]
    fitted.assimilate("lobith", t1, "cd", before * 1.5, False, features=f)
    after = fitted.predict("lobith", t1, f)
    assert before < after.median()["cd"] < before * 1.5
    assert after.sd_log()["cd"] < fitted.predict("lobith", t1 - timedelta(seconds=1), f, as_of=t1).sd_log()["cd"]
    # cross-metal: zn moves too if cd and zn levels are correlated; at least it must stay finite
    assert np.isfinite(after.median()["zn"])


def test_out_of_order_equals_in_order(table):
    f = {"q_mean_7d": 1500}
    t1, t2 = CUT + timedelta(days=10), CUT + timedelta(days=24)
    obs = [(t1, "cd", 2.0), (t1, "zn", 500.0), (t2, "cd", 1.0)]
    preds = []
    for order in (obs, obs[::-1]):
        k = KalmanSoftSensor(METALS, FEATS)
        k.fit(table, CUT)
        for t, m, v in order:
            k.assimilate("lobith", t, m, v, False, features=f)
        preds.append(k.predict("lobith", t2 + timedelta(days=5), f).mean_log)
    assert np.allclose(preds[0], preds[1])


def test_as_of_hides_late_results(fitted):
    f = {"q_mean_7d": 1500}
    t_s = CUT + timedelta(days=10)
    t_pred = CUT + timedelta(days=15)
    base = fitted.predict("lobith", t_pred, f).mean_log
    fitted.assimilate("lobith", t_s, "cd", 5.0, False, features=f, available_utc=t_s + timedelta(days=14))
    # at t_pred the result has not arrived yet -> same as without it
    assert np.allclose(fitted.predict("lobith", t_pred, f, as_of=t_pred).mean_log, base)
    assert not np.allclose(fitted.predict("lobith", t_pred, f, as_of=t_s + timedelta(days=15)).mean_log, base)


def test_outlier_is_limited(fitted):
    f = {"q_mean_7d": 1500}
    t1 = CUT + timedelta(days=30)
    prior = fitted.predict("lobith", t1, f)
    fitted.assimilate("lobith", t1, "cd", prior.median()["cd"] * 100, False, features=f)  # absurd value
    post = fitted.predict("lobith", t1, f)
    # Huber update: the level moves by at most ~outlier_k prior sds, far less than ln(100)
    assert post.mean_log[0] - prior.mean_log[0] < 0.5 * np.log(100)


def test_one_step_noise_fit_never_worse_on_training(table):
    k = KalmanSoftSensor(METALS, FEATS, noise_fit="one_step")
    k.fit(table, CUT)
    scores = k.diagnostics["lobith"]["q_multiplier_score"]
    assert scores.min() <= scores.loc[1.0]                     # the grid includes the variogram Q
    assert k.predict("lobith", CUT, {"q_mean_7d": 1500}).mean_log.shape == (2,)
    with pytest.raises(ValueError, match="noise_fit"):
        KalmanSoftSensor(METALS, FEATS, noise_fit="other")
