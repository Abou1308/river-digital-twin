"""Are the differences between baseline models real? Significance tests that respect autocorrelation.

Setup: fit on training (2012-2020), predict every eval_primary sample (2021-2024).
eval_secondary (2025) is not used. Run for several assumed lab turnaround times (delays), because
the "last lab value" features depend on when a lab result becomes available.

Models
  median    training-period median (constant)
  last      last lab value already known at prediction time
  ridge_s   ridge on sensor features only
  ridge_sl  ridge on sensor features + last known lab value + its age   <- reference model
  gbm_sl    gradient boosting on the same inputs as ridge_sl

Loss per sample: absolute error of the log concentration, |ln pred - ln actual| (≈ relative error,
symmetric for over- and under-prediction). Medians of absolute % error are shown for readability.

Tests of ridge_sl against each other model, on the per-sample loss difference d_t = L_other - L_ref
(positive = reference model better):
  DM   Diebold-Mariano test of mean(d) = 0 with Newey-West (HAC) variance and the Harvey-Leybourne-
       Newbold small-sample correction. HAC variance accounts for errors of neighbouring samples
       being correlated.
  boot circular block bootstrap (blocks of ~2 months of samples) 95% interval for the difference in
       median absolute % error. Blocks keep neighbouring samples together.

Usage: python analysis/03_model_comparison.py data/raw/dataset/Full/model_table.csv
"""
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
m = pd.read_csv(sys.argv[1], parse_dates=["sample_time_utc"])
METALS = ["cd", "cu", "ni", "pb", "zn"]
SENSOR = [c for c in m.columns if c.startswith(("turb_", "q_", "spm_"))] + ["doy_sin", "doy_cos"]
DELAYS = [0, 7, 14, 28]
BLOCK = {"eijsden": 8, "lobith": 4}  # samples per block, about 2 months at both stations
N_BOOT = 2000
REF = "ridge_sl"
OTHERS = ["last", "ridge_s", "gbm_sl", "median"]
rng = np.random.default_rng(0)


def known_lab(g, k, delay_d):
    """Most recent lab value of metal k known at each sample time, if results arrive delay_d days late."""
    g = g.sort_values("sample_time_utc")
    lab = g[["sample_time_utc", k]].dropna().rename(columns={"sample_time_utc": "lab_time", k: "known"})
    lab["available_at"] = lab.lab_time + pd.Timedelta(days=delay_d)
    out = pd.merge_asof(g[["sample_time_utc"]], lab.sort_values("available_at"), left_on="sample_time_utc",
                        right_on="available_at", allow_exact_matches=False).set_index(g.index)
    return np.log(out.known), (g.sample_time_utc - out.lab_time).dt.total_seconds() / 86400


def ridge():
    return make_pipeline(SimpleImputer(), StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))


def dm_test(d, lag):
    """Diebold-Mariano with Newey-West variance and HLN correction. Returns (stat, two-sided p)."""
    d = np.asarray(d, float)
    n = len(d)
    dc = d - d.mean()
    gamma = [np.dot(dc[: n - j], dc[j:]) / n for j in range(lag + 1)]
    lrv = gamma[0] + 2 * sum((1 - j / (lag + 1)) * gamma[j] for j in range(1, lag + 1))
    if lrv <= 0:
        lrv = gamma[0]
    dm = d.mean() / np.sqrt(lrv / n)
    h = lag + 1
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = dm * hln
    return stat, 2 * stats.t.sf(abs(stat), n - 1)


def block_boot_median_diff(ape_other, ape_ref, block):
    """Circular block bootstrap of median(APE_other) - median(APE_ref). Returns 95% interval."""
    n = len(ape_ref)
    nb = int(np.ceil(n / block))
    diffs = np.empty(N_BOOT)
    for b in range(N_BOOT):
        starts = rng.integers(0, n, nb)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n] % n
        diffs[b] = np.median(ape_other[idx]) - np.median(ape_ref[idx])
    return np.percentile(diffs, [2.5, 97.5])


rows, errs = [], []
for st, g in m.groupby("station"):
    g = g.sort_values("sample_time_utc")
    for k in METALS:
        for delay in DELAYS:
            prev, age = known_lab(g, k, delay)
            d = g.assign(prev_known=prev, prev_age_d=age).dropna(subset=[k, "prev_known"])
            tr, te = d[d.period == "training"], d[d.period == "eval_primary"]
            y, yt = np.log(tr[k]), np.log(te[k]).values
            X = SENSOR + ["prev_known", "prev_age_d"]
            pred = {
                "median": np.full(len(te), y.median()),
                "last": te.prev_known.values,
                "ridge_s": ridge().fit(tr[SENSOR], y).predict(te[SENSOR]),
                "ridge_sl": ridge().fit(tr[X], y).predict(te[X]),
                "gbm_sl": HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=3,
                                                        min_samples_leaf=20).fit(tr[X], y).predict(te[X]),
            }
            loss = {n_: np.abs(p - yt) for n_, p in pred.items()}
            ape = {n_: np.abs(np.exp(p) - np.exp(yt)) / np.exp(yt) * 100 for n_, p in pred.items()}
            for n_ in pred:
                errs.append(dict(station=st, metal=k, delay_d=delay, model=n_, median_ape=np.median(ape[n_])))
            for other in OTHERS:
                stat, p = dm_test(loss[other] - loss[REF], lag=BLOCK[st] - 1)
                lo, hi = block_boot_median_diff(ape[other], ape[REF], BLOCK[st])
                rows.append(dict(station=st, metal=k, delay_d=delay, vs=other, n=len(te),
                                 ref_median_ape=np.median(ape[REF]), other_median_ape=np.median(ape[other]),
                                 diff=np.median(ape[other]) - np.median(ape[REF]), boot_lo=lo, boot_hi=hi,
                                 dm_stat=stat, dm_p=p))

res = pd.DataFrame(rows)
E = pd.DataFrame(errs)
pd.set_option("display.width", 220)

print("Median absolute % error, eval_primary 2021-2024 (rows: station/metal, columns: delay in days)\n")
print(E.pivot_table(index=["station", "metal", "model"], columns="delay_d", values="median_ape").round(1)
      .unstack("model").swaplevel(axis=1).sort_index(axis=1)
      .reindex(columns=["median", "last", "ridge_s", "ridge_sl", "gbm_sl"], level=0).to_string())


def verdict(r):
    if r.dm_p < 0.05 and r.boot_lo > 0:
        return "ridge_sl better"
    if r.dm_p < 0.05 and r.boot_hi < 0:
        return "ridge_sl worse"
    if r.dm_p < 0.05:
        return "DM only"
    return "no sig. difference"


res["verdict"] = res.apply(verdict, axis=1)
out = res.copy()
for c in ["ref_median_ape", "other_median_ape", "diff", "boot_lo", "boot_hi", "dm_stat"]:
    out[c] = out[c].round(1)
out["dm_p"] = out.dm_p.map(lambda v: f"{v:.1e}" if v < 1e-3 else f"{v:.3f}")
print("\n\nridge_sl (sensors + last known lab value) vs each other model")
print("diff = other median % error - ridge_sl median % error (positive = ridge_sl better); "
      "boot_lo/hi = 95% block-bootstrap interval of diff\n")
for other in OTHERS:
    print(f"--- vs {other}")
    print(out[out.vs == other].drop(columns="vs").to_string(index=False))
    print()

print("Summary: count of station/metal combinations (out of 10) per verdict")
print(res.pivot_table(index=["vs", "delay_d"], columns="verdict", values="n", aggfunc="count", fill_value=0)
      .to_string())

print("\nHolm-corrected (10 station/metal tests per comparison and delay), DM test:")
for (vs, delay), g in res.groupby(["vs", "delay_d"]):
    ps = np.sort(g.dm_p.values)
    k = int((ps < 0.05 / (len(ps) - np.arange(len(ps)))).cumprod().sum())
    print(f"  vs {vs:8s} delay {delay:2d} d: {k} / {len(ps)} significant")
