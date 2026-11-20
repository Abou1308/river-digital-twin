"""Online recalibration of event probabilities (Platt scaling), learned from the outcomes received so far.

A forecast probability p is replaced by sigmoid(a + b * logit(p)). The coefficients are fitted by
logistic regression on the earlier forecasts whose outcome had arrived before the forecast is made, so
nothing from the future is used. The fit starts from "the forecasts are already calibrated" (a = 0,
b = 1): calibrated pseudo-forecasts at 0.1, 0.3, 0.5, 0.7 and 0.9, each with an outcome equal to its
forecast, are added with a total weight of `prior_n` samples, so a few early outcomes cannot swing the
correction and the fit is defined even with one outcome.

b < 1 pulls overconfident forecasts towards 0.5; a shifts them up or down. Used in
analysis/07_calibration_options.py.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-4
_PRIOR_P = np.array([0.1, 0.3, 0.5, 0.7, 0.9])


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def fit_platt(p: np.ndarray, outcome: np.ndarray, weight: np.ndarray, n_iter: int = 50) -> tuple[float, float]:
    """Weighted logistic regression of outcome (0..1, may be fractional) on logit(p); returns (a, b)."""
    X = np.column_stack([np.ones(len(p)), _logit(np.asarray(p, float))])
    y, w = np.asarray(outcome, float), np.asarray(weight, float)
    def loglik(b: np.ndarray) -> float:
        eta = X @ b
        return float(np.sum(w * (y * eta - np.logaddexp(0, eta))))

    # Newton's method with step halving: a full Newton step can overshoot and diverge when the
    # optimum is far from the starting point (a = 0, b = 1).
    beta = np.array([0.0, 1.0])
    for _ in range(n_iter):
        mu = 1 / (1 + np.exp(-X @ beta))
        grad = X.T @ (w * (y - mu))
        hess = (X * (w * mu * (1 - mu))[:, None]).T @ X
        step = np.linalg.solve(hess, grad)
        current = loglik(beta)
        while loglik(beta + step) < current and np.max(np.abs(step)) > 1e-10:
            step = step / 2
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    return float(beta[0]), float(beta[1])


def online_platt(p: np.ndarray, outcome: np.ndarray, forecast_time: np.ndarray, available_time: np.ndarray,
                 prior_n: float) -> np.ndarray:
    """Recalibrated probabilities, each fitted only on outcomes available before its forecast time."""
    p, o = np.asarray(p, float), np.asarray(outcome, float)
    forecast_time, available_time = np.asarray(forecast_time), np.asarray(available_time)
    out = np.empty_like(p)
    for i in range(len(p)):
        past = np.flatnonzero(available_time < forecast_time[i])
        if len(past) == 0:
            out[i] = p[i]
            continue
        pp = np.concatenate([p[past], _PRIOR_P])
        yy = np.concatenate([o[past], _PRIOR_P])
        ww = np.concatenate([np.ones(len(past)), np.full(len(_PRIOR_P), prior_n / len(_PRIOR_P))])
        a, b = fit_platt(pp, yy, ww)
        out[i] = 1 / (1 + np.exp(-(a + b * _logit(p[i:i + 1])[0])))
    return out
