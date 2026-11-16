"""Long-term trend test for metal concentrations: seasonal Mann-Kendall with Sen's slope.

Method
------
1. Each station/metal series is reduced to one value per calendar month per year (monthly median of
   log concentration). This is the usual input for a seasonal trend test and removes the effect of
   having several samples in some months.
2. Seasonal Kendall test (Hirsch, Slack & Smith 1982): a Mann-Kendall S statistic is computed within
   each calendar month across years (January 2012 vs January 2013 ...), so seasonality cannot create
   a false trend. The 12 S values are summed.
3. Variance with the Hirsch & Slack (1984) covariance correction: months are not independent of each
   other (a high-level year is high in every month), so the variance of the summed S includes the
   covariances between months. This is the correction for serial dependence.
4. Seasonal Sen slope: median of all within-month slopes between years, on log scale, reported as
   % change per year with a 95% interval (from the same variance).

Two runs per series:
  training  2012-2020  - this is what may inform model design (no peek at the evaluation years)
  all       2012-2025  - descriptive, for the report

Usage: python analysis/02_trend_test.py data/raw/dataset/Full/model_table.csv
"""
import sys

import numpy as np
import pandas as pd
from scipy import stats

m = pd.read_csv(sys.argv[1], parse_dates=["sample_time_utc"])
m["year"] = m.sample_time_utc.dt.year
m["month"] = m.sample_time_utc.dt.month
METALS = ["cd", "cu", "ni", "pb", "zn"]


def sgn(x):
    return np.sign(x)


def seasonal_mk(table):
    """table: DataFrame years x months (log values, NaN = missing). Returns S, var (Hirsch-Slack), z, p."""
    X = table.values  # n years x 12 months
    n, p = X.shape
    S = np.zeros(p)
    var_indep = np.zeros(p)
    for g in range(p):
        x = X[:, g]
        obs = ~np.isnan(x)
        xv = x[obs]
        k = len(xv)
        if k < 3:
            continue
        S[g] = sum(sgn(xv[j] - xv[i]) for i in range(k - 1) for j in range(i + 1, k))
        _, ties = np.unique(xv, return_counts=True)
        var_indep[g] = (k * (k - 1) * (2 * k + 5) - np.sum(ties * (ties - 1) * (2 * ties + 5))) / 18
    # Hirsch & Slack covariance between seasons g and h (missing values: rank (n+1)/2, sign 0)
    R = np.full_like(X, np.nan)
    for g in range(p):
        col = X[:, g]
        obs = ~np.isnan(col)
        r = np.full(n, (n + 1) / 2)
        r[obs] = stats.rankdata(col[obs]) + (n - obs.sum()) / 2  # centre ranks when values are missing
        R[:, g] = r
    cov = np.zeros((p, p))
    for g in range(p):
        for h in range(g + 1, p):
            K = 0.0
            for i in range(n - 1):
                for j in range(i + 1, n):
                    a = X[j, g] - X[i, g]
                    b = X[j, h] - X[i, h]
                    if not (np.isnan(a) or np.isnan(b)):
                        K += sgn(a * b)
            cov[g, h] = cov[h, g] = (K + 4 * np.sum(R[:, g] * R[:, h]) - n * (n + 1) ** 2) / 3
    S_tot = S.sum()
    var = var_indep.sum() + cov.sum()
    var = max(var, var_indep.sum() * 0.1)  # guard against a negative estimate with very few years
    z = (S_tot - np.sign(S_tot)) / np.sqrt(var) if S_tot != 0 else 0.0
    p_val = 2 * stats.norm.sf(abs(z))
    return S_tot, var_indep.sum(), var, z, p_val


def seasonal_sen(table, var):
    X = table.values
    years = table.index.values.astype(float)
    slopes = []
    for g in range(X.shape[1]):
        x = X[:, g]
        for i in range(len(x) - 1):
            for j in range(i + 1, len(x)):
                if not (np.isnan(x[i]) or np.isnan(x[j])):
                    slopes.append((x[j] - x[i]) / (years[j] - years[i]))
    slopes = np.sort(slopes)
    N = len(slopes)
    C = 1.96 * np.sqrt(var)
    lo_i = int(np.clip(np.floor((N - C) / 2), 0, N - 1))
    hi_i = int(np.clip(np.ceil((N + C) / 2), 0, N - 1))
    return np.median(slopes), slopes[lo_i], slopes[hi_i]


def pct(b):
    return (np.exp(b) - 1) * 100


rows = []
for label, sel in [("training 2012-2020", m.period == "training"), ("all 2012-2025", m.period.notna())]:
    for st, g in m[sel].groupby("station"):
        for k in METALS:
            d = g.dropna(subset=[k])
            table = (d.assign(lv=np.log(d[k])).groupby(["year", "month"]).lv.median().unstack("month")
                     .reindex(columns=range(1, 13)))
            S, v0, v, z, p = seasonal_mk(table)
            p_naive = 2 * stats.norm.sf(abs((S - np.sign(S)) / np.sqrt(v0))) if S else 1.0
            b, lo, hi = seasonal_sen(table, v)
            first, last = table.index.min(), table.index.max()
            rows.append(dict(period=label, station=st, metal=k, years=f"{first}-{last}",
                             sen_pct_per_year=pct(b), ci95_lo=pct(lo), ci95_hi=pct(hi),
                             total_change_pct=pct(b * (last - first)),
                             p_ignoring_dependence=p_naive, p_hirsch_slack=p))

res = pd.DataFrame(rows)
pd.set_option("display.width", 200)
out = res.copy()
for c in ["sen_pct_per_year", "ci95_lo", "ci95_hi", "total_change_pct"]:
    out[c] = out[c].round(1)
for c in ["p_ignoring_dependence", "p_hirsch_slack"]:
    out[c] = out[c].map(lambda v: f"{v:.1e}" if v < 1e-3 else f"{v:.3f}")
for label, g in out.groupby("period", sort=False):
    print(f"\n=== {label}: seasonal Mann-Kendall on monthly medians of log concentration")
    print(g.drop(columns="period").to_string(index=False))
    r = res[res.period == label]
    ps = np.sort(r.p_hirsch_slack.values)
    holm = int((ps < 0.05 / (len(ps) - np.arange(len(ps)))).cumprod().sum())
    print(f"significant at 0.05 (Hirsch-Slack): {(r.p_hirsch_slack < 0.05).sum()} / {len(r)}; "
          f"after Holm correction for {len(r)} tests: {holm}")
