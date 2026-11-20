"""Main soft sensor: dynamic regression in state-space form, run as a Kalman filter.

Model, per station, in log space (m metals, k sensor inputs):

    observation:  y_s = level_s + B x_s + v_s,          v_s ~ N(0, R)      (at lab samples only)
    state:        level_t = level_s + w,                 w   ~ N(0, Q * (t - s) [days])

    y_s   ln(metal concentrations) of the lab sample taken at time s (any subset of metals)
    x_s   transformed sensor inputs at s (ln of window means of discharge / turbidity / SPM, season)
    B     (m x k) dilution / season coefficients
    level slowly varying level per metal (random walk): absorbs long-term change such as the Eijsden
          Cd decline, which the training years do not show (analysis/02_trend_test.py)
    Q     (m x m) level drift per day, full matrix: a Zn result also informs Cd and Pb
    R     (m x m) sample-level variation not explained by level + B x (lab error + short-term variation)

Fitting (training period only):
  1. B from ridge regression of *changes* between consecutive lab results on changes in x. Differencing
     removes the slowly varying level, so drift cannot bias B (regression with ARIMA(0,1,0) errors).
  2. Q and R from the variogram of the residuals e = y - B x: for two samples d days apart,
     Cov(e_j - e_i) = 2R + Q d. Pair covariances are binned by lag and a straight line is fitted per
     metal pair (intercept -> R, slope -> Q). Robust (MAD-based) covariances limit the influence of
     outliers. `diagnostics[station]` keeps the binned variogram so the linearity can be checked.
  3. The filter is run through the training samples to get the state at the end of training.

Live:
  predict    -> level uncertainty grows with the time since the last result; mean = level + B x_t;
                covariance = P + Q dt + R (the spread a new lab sample would show).
  assimilate -> measurement update at the *sample* time. Lab results arrive late (turnaround) and can
                arrive out of order, so results are kept sorted by sample time and the filter is
                re-run from the affected sample (cheap: a few hundred samples per station).
  as_of      -> predictions can be restricted to results that had arrived before a given moment:
                that is what the twin could actually have known then (used for fair evaluation).
Automatic robustness (applied identically in training and live):
  - outliers: if a result is more than `outlier_k` predicted sd away, its noise variance is inflated
    so that it sits at outlier_k sd (Huber-type update) instead of pulling the level fully.
  - prediction intervals are widened per metal by a factor calibrated on training (heavy tails).
  - censored results (value = detection limit): ignored if the prediction is already below the
    limit, otherwise used with 4x noise variance. (Metals in this dataset are never censored.)
"""

from __future__ import annotations

import bisect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from rdt.softsensor.base import FeatureTransform, LabObservation, NotReadyError, SoftSensor, check_no_leakage
from rdt.softsensor.baselines import _psd
from rdt.softsensor.estimate import MetalEstimate

DAY_S = 86400.0


@dataclass
class _Params:
    tf: FeatureTransform
    B: np.ndarray       # (m, k)
    Q: np.ndarray       # (m, m) per day
    R: np.ndarray       # (m, m)
    scale: np.ndarray | None = None   # (m,) interval calibration factors, see fit


@dataclass
class _State:
    t: pd.Timestamp
    a: np.ndarray       # level mean (m,)
    P: np.ndarray       # level covariance (m, m)


@dataclass
class _Group:
    """All results of one lab sample (same station and sample time)."""

    t: pd.Timestamp
    obs: dict[str, LabObservation]
    x: np.ndarray


def _robust_scale(u: np.ndarray) -> float:
    return 1.4826 * float(np.median(np.abs(u - np.median(u))))


def _robust_cov(u: np.ndarray, v: np.ndarray) -> float:
    """Gnanadesikan-Kettenring covariance with MAD scales."""
    su, sv = _robust_scale(u), _robust_scale(v)
    if su == 0 or sv == 0:
        return 0.0
    a, b = u / su, v / sv
    return su * sv * (_robust_scale(a + b) ** 2 - _robust_scale(a - b) ** 2) / 4


class KalmanSoftSensor(SoftSensor):
    name = "kalman"

    def __init__(self, metals: Sequence[str], features: Sequence[str] = (), max_lag_d: float = 180.0,
                 n_bins: int = 12, outlier_k: float = 3.5) -> None:
        super().__init__(metals, features)
        self.max_lag_d, self.n_bins, self.outlier_k = max_lag_d, n_bins, outlier_k
        self._params: dict[str, _Params] = {}
        self._init: dict[str, _State] = {}
        self._groups: dict[str, list[_Group]] = {}
        self._ckpt: dict[str, list[_State]] = {}
        self.diagnostics: dict[str, dict[str, object]] = {}

    # ------------------------------------------------------------------------------------- fit
    def fit(self, train: pd.DataFrame, cutoff: pd.Timestamp) -> None:
        check_no_leakage(train, cutoff)
        for st, g in train.sort_values("sample_time_utc").groupby("station"):
            g = g.reset_index(drop=True)
            tf = FeatureTransform(self.features).fit(g)
            X = tf.transform(g)
            Yraw = g[list(self.metals)].to_numpy(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                Y = np.log(np.where(Yraw > 0, Yraw, np.nan))
            B = self._fit_B(X, Y)
            E = Y - X @ B.T
            t_d = (g["sample_time_utc"] - g["sample_time_utc"].iloc[0]).dt.total_seconds().to_numpy() / DAY_S
            Q, R, diag = self._fit_noise(t_d, E)
            p = _Params(tf, B, Q, R)
            self._params[st] = p

            state, z_train = self._run_training(p, g, X, Y, Yraw, E)
            self._init[st] = state
            self._groups[st], self._ckpt[st] = [], []
            # Interval calibration: the robust (MAD) variances describe the bulk of the data, but lab
            # results have heavier tails, so a +-1.645 sd interval covers <90% of training results.
            # Per metal, widen the reported covariance so that 90% of training innovations (after a
            # burn-in of 20 samples) fall inside. Only the uncertainty changes, not the predictions.
            Z = pd.DataFrame(z_train, columns=self.metals)
            q90 = Z.iloc[20:].abs().quantile(0.9).to_numpy()
            p.scale = np.where(np.isfinite(q90), np.maximum(1.0, q90 / 1.6448536269514722), 1.0)
            diag["B"] = pd.DataFrame(B, index=self.metals, columns=self.features)
            diag["train_innovation_z"] = Z
            diag["interval_scale"] = pd.Series(p.scale, index=self.metals)
            self.diagnostics[st] = diag

    def _run_training(self, p: _Params, g: pd.DataFrame, X: np.ndarray, Y: np.ndarray, Yraw: np.ndarray,
                      E: np.ndarray) -> tuple[_State, list[np.ndarray]]:
        """Run the filter through the training samples; return the state at the end of training and the
        standardised innovations per sample."""
        a0 = np.nanmedian(E[: min(10, len(E))], axis=0)
        a0 = np.where(np.isnan(a0), np.nanmedian(E, axis=0), a0)
        state = _State(g["sample_time_utc"].iloc[0], a0, np.diag(np.nanvar(E, axis=0)))
        z_train = []
        for i, row in g.iterrows():
            obs = {m: (Yraw[i, j], False) for j, m in enumerate(self.metals) if np.isfinite(Y[i, j])}
            state, z = self._update(p, state, row["sample_time_utc"], obs, X[i])
            z_train.append(z)
        return state, z_train

    def _fit_B(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        B = np.zeros((Y.shape[1], X.shape[1]))
        for j in range(Y.shape[1]):
            ok = np.flatnonzero(np.isfinite(Y[:, j]))
            if len(ok) < 10:
                continue
            dY, dX = np.diff(Y[ok, j]), np.diff(X[ok], axis=0)
            B[j] = RidgeCV(alphas=np.logspace(-2, 4, 25), fit_intercept=False).fit(dX, dY).coef_
        return B

    def _fit_noise(self, t_d: np.ndarray, E: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        m = E.shape[1]
        ii, jj = np.triu_indices(len(t_d), 1)
        lag = t_d[jj] - t_d[ii]
        keep = (lag > 0) & (lag <= self.max_lag_d)
        ii, jj, lag = ii[keep], jj[keep], lag[keep]
        D = E[jj] - E[ii]
        edges = np.linspace(0, self.max_lag_d, self.n_bins + 1)
        b = np.clip(np.digitize(lag, edges) - 1, 0, self.n_bins - 1)
        gam = np.full((self.n_bins, m, m), np.nan)
        cnt = np.zeros((self.n_bins, m, m))
        mid = np.array([lag[b == k].mean() if np.any(b == k) else np.nan for k in range(self.n_bins)])
        for k in range(self.n_bins):
            Dk = D[b == k]
            for p in range(m):
                for q in range(p, m):
                    ok = np.isfinite(Dk[:, p]) & np.isfinite(Dk[:, q])
                    if ok.sum() < 20:
                        continue
                    c = _robust_scale(Dk[ok, p]) ** 2 if p == q else _robust_cov(Dk[ok, p], Dk[ok, q])
                    gam[k, p, q] = gam[k, q, p] = c
                    cnt[k, p, q] = cnt[k, q, p] = ok.sum()
        C0, Qm = np.zeros((m, m)), np.zeros((m, m))
        for p in range(m):
            for q in range(p, m):
                ok = np.isfinite(gam[:, p, q]) & np.isfinite(mid)
                if ok.sum() < 3:
                    continue
                A = np.column_stack([np.ones(ok.sum()), mid[ok]])
                w = np.sqrt(cnt[ok, p, q])
                coef = np.linalg.lstsq(A * w[:, None], gam[ok, p, q] * w, rcond=None)[0]
                C0[p, q] = C0[q, p] = coef[0]
                Qm[p, q] = Qm[q, p] = coef[1]
        # a negative fitted slope / intercept means "no detectable drift / noise": keep a small floor
        R = _psd(C0 / 2, floor=1e-4)
        Q = _psd(Qm, floor=1e-7)
        diag = {
            "variogram": {mt: pd.DataFrame({"lag_d": mid, "gamma": gam[:, j, j], "fit": C0[j, j] + Qm[j, j] * mid,
                                            "pairs": cnt[:, j, j]}) for j, mt in enumerate(self.metals)},
            "Q": pd.DataFrame(Q, index=self.metals, columns=self.metals),
            "R": pd.DataFrame(R, index=self.metals, columns=self.metals),
        }
        return Q, R, diag

    # ---------------------------------------------------------------------------- filter step
    def _update(self, p: _Params, s: _State, t: pd.Timestamp, obs: Mapping[str, tuple[float, bool]],
                x: np.ndarray) -> tuple[_State, np.ndarray]:
        """Time update to t, then measurement update with obs {metal: (value, censored)}.
        Returns the new state and the standardised innovations (NaN for unobserved metals)."""
        m = len(self.metals)
        dt = max(0.0, (t - s.t).total_seconds() / DAY_S)
        a, P = s.a, s.P + p.Q * dt
        pred = a + p.B @ x
        idx, y, cens = [], [], []
        for metal, (v, c) in obs.items():
            j = self.metals.index(metal)
            lv = float(np.log(v))
            if c and pred[j] < lv:          # below the detection limit, as predicted: nothing to learn
                continue
            idx.append(j)
            y.append(lv)
            cens.append(c)
        zstd = np.full(m, np.nan)
        if not idx:
            return _State(t, a, P), zstd
        ix = np.array(idx)
        z = np.array(y) - pred[ix]
        Reff = p.R[np.ix_(ix, ix)].copy()
        S0 = P[np.ix_(ix, ix)] + Reff
        zstd[ix] = z / np.sqrt(np.diag(S0))
        for k in range(len(ix)):
            if cens[k]:
                Reff[k, k] *= 4
            sd2 = P[ix[k], ix[k]] + Reff[k, k]
            if z[k] ** 2 > self.outlier_k ** 2 * sd2:
                Reff[k, k] += z[k] ** 2 / self.outlier_k ** 2 - sd2
        S = P[np.ix_(ix, ix)] + Reff
        K = np.linalg.solve(S, P[ix, :]).T            # = P[:, ix] S^-1  (S, P symmetric)
        a = a + K @ z
        P = P - K @ P[ix, :]
        return _State(t, a, (P + P.T) / 2), zstd

    # --------------------------------------------------------------------------- live state
    def _on_observation(self, station: str, obs: LabObservation) -> None:
        if station not in self._params:
            return
        groups = self._groups.setdefault(station, [])
        times = [g.t for g in groups]
        k = bisect.bisect_left(times, obs.sample_time)
        if k < len(groups) and groups[k].t == obs.sample_time:
            groups[k].obs[obs.metal] = obs
        else:
            groups.insert(k, _Group(obs.sample_time, {obs.metal: obs},
                                    self._params[station].tf.transform_one(obs.features)))
        del self._ckpt[station][k:]                  # re-run the filter from this sample on

    def _extend(self, station: str, upto: int) -> _State:
        """Checkpoints for groups[:upto] (all results known); returns the state after them."""
        p, groups, ckpt = self._params[station], self._groups[station], self._ckpt[station]
        state = ckpt[-1] if ckpt else self._init[station]
        for k in range(len(ckpt), upto):
            g = groups[k]
            state, _ = self._update(p, state, g.t, {m: (o.value, o.censored) for m, o in g.obs.items()}, g.x)
            ckpt.append(state)
        return ckpt[upto - 1] if upto > 0 else self._init[station]

    def _state(self, station: str, as_of: datetime | None) -> _State:
        groups = self._groups.get(station, [])
        if as_of is None:
            return self._extend(station, len(groups))
        cut = pd.Timestamp(as_of)
        first = next((k for k, g in enumerate(groups) if any(o.available >= cut for o in g.obs.values())),
                     len(groups))
        state = self._extend(station, first)
        p = self._params[station]
        for g in groups[first:]:
            known = {m: (o.value, o.censored) for m, o in g.obs.items() if o.available < cut}
            if known:
                state, _ = self._update(p, state, g.t, known, g.x)
        return state

    def predict(self, station: str, t: datetime, features: Mapping[str, float],
                as_of: datetime | None = None) -> MetalEstimate:
        if station not in self._params:
            raise NotReadyError(f"No state for station {station}")
        p = self._params[station]
        s = self._state(station, as_of)
        dt = max(0.0, (pd.Timestamp(t) - s.t).total_seconds() / DAY_S)
        mean = s.a + p.B @ p.tf.transform_one(features)
        cov = s.P + p.Q * dt + p.R
        if p.scale is not None:
            cov = cov * np.outer(p.scale, p.scale)
        return MetalEstimate(station, t, self.metals, mean, (cov + cov.T) / 2, model=self.name)
