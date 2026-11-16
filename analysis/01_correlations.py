"""Metal-sensor Spearman correlations with autocorrelation- and seasonality-aware significance.

Naive Spearman p-values assume independent samples. Lab samples here are ~weekly (Eijsden) or
~biweekly (Lobith) and strongly autocorrelated, and both metals and sensors are seasonal. This
script reports, per station / metal / sensor feature:

  rho, p_naive          ordinary Spearman (assumes independence: too optimistic)
  n_eff, p_neff         effective sample size from the rank autocorrelations of both series
                        (Pyper & Peterman 1998: 1/n_eff = 1/n + 2/n * sum_j (n-j)/n * r_xx(j) r_yy(j),
                        j = 1..n/5), t-test with n_eff - 2 df
  p_shift               circular-shift test: the sensor series is shifted against the metal series by
                        every offset of at least one year; p = share of shifts with |rho| >= observed.
                        Keeps each series' own autocorrelation and seasonality.
  rho_anom, p_anom_neff Spearman of monthly anomalies (each value minus its station's median for that
                        calendar month), with the same n_eff correction. Answers: does the relation
                        hold beyond the shared seasonal cycle?

Samples are treated as an ordered sequence (sampling is near-regular: 90% of gaps are exactly 7 / 14 d).

Usage: python analysis/01_correlations.py data/raw/dataset/Full/model_table.csv [period]

period (optional): e.g. "training" to use only 2012-2020. Use "training" for anything that informs
feature selection, so the evaluation years (2021-2025) stay unseen.
"""
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
m = pd.read_csv(sys.argv[1], parse_dates=["sample_time_utc"]).sort_values("sample_time_utc")
PERIOD = sys.argv[2] if len(sys.argv) > 2 else None
if PERIOD:
    m = m[m.period == PERIOD]
print(f"period: {PERIOD or 'all'}  ({m.sample_time_utc.min():%Y-%m-%d} to {m.sample_time_utc.max():%Y-%m-%d}, "
      f"{len(m)} samples)\n")
m["month"] = m.sample_time_utc.dt.month
METALS = ["cd", "cu", "ni", "pb", "zn"]
FEATS = ["q_mean_7d", "turb_mean_7d", "spm_mean_7d"]
PER_YEAR = {"eijsden": 52, "lobith": 26}


def acf(x, max_lag):
    x = np.asarray(x, float) - np.mean(x)
    v = np.dot(x, x)
    return np.array([np.dot(x[:-j], x[j:]) / v for j in range(1, max_lag + 1)])


def n_eff(x, y):
    n = len(x)
    L = max(1, n // 5)
    rx, ry = acf(stats.rankdata(x), L), acf(stats.rankdata(y), L)
    w = (n - np.arange(1, L + 1)) / n
    inv = 1 / n + 2 / n * np.sum(w * rx * ry)
    return float(np.clip(1 / inv, 3, n))


def p_from_r(r, n):
    t = r * np.sqrt((n - 2) / max(1e-12, 1 - r * r))
    return 2 * stats.t.sf(abs(t), n - 2)


def shift_test(x, y, min_shift):
    n = len(x)
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    obs = abs(np.corrcoef(rx, ry)[0, 1])
    shifts = range(min_shift, n - min_shift + 1)
    null = np.array([abs(np.corrcoef(rx, np.roll(ry, s))[0, 1]) for s in shifts])
    return (1 + np.sum(null >= obs)) / (1 + len(null)), len(null)


rows = []
for st, g in m.groupby("station"):
    for k in METALS:
        for f in FEATS:
            d = g[["sample_time_utc", "month", k, f]].dropna()
            d = d[(d[k] > 0) & (d[f] > 0)]
            x, y = np.log(d[k].values), np.log(d[f].values)
            n = len(d)
            rho, p_naive = stats.spearmanr(x, y)
            ne = n_eff(x, y)
            p_shift, n_shifts = shift_test(x, y, PER_YEAR[st])
            # monthly anomalies (log scale)
            ax = x - d.groupby("month")[k].transform(lambda s: np.log(s).median()).values
            ay = y - d.groupby("month")[f].transform(lambda s: np.log(s).median()).values
            rho_a = stats.spearmanr(ax, ay)[0]
            ne_a = n_eff(ax, ay)
            rows.append(dict(station=st, metal=k, feature=f, n=n, rho=rho, p_naive=p_naive,
                             n_eff=ne, p_neff=p_from_r(rho, ne), p_shift=p_shift,
                             rho_anom=rho_a, n_eff_anom=ne_a, p_anom_neff=p_from_r(rho_a, ne_a)))

res = pd.DataFrame(rows)
pd.set_option("display.width", 220)
fmt = res.copy()
for c in ["rho", "rho_anom"]:
    fmt[c] = fmt[c].round(2)
for c in ["n_eff", "n_eff_anom"]:
    fmt[c] = fmt[c].round(0).astype(int)
for c in ["p_naive", "p_neff", "p_shift", "p_anom_neff"]:
    fmt[c] = fmt[c].map(lambda v: f"{v:.1e}" if v < 1e-3 else f"{v:.3f}")
print(fmt.to_string(index=False))

print("\nNumber of the 30 pairs significant at p < 0.05 under each method:")
for c in ["p_naive", "p_neff", "p_shift", "p_anom_neff"]:
    print(f"  {c:12s} {int((res[c] < 0.05).sum()):2d} / {len(res)}")
print("\nWith Holm correction for 30 tests (p_neff, p_anom_neff):")
for c in ["p_neff", "p_anom_neff"]:
    p_sorted = np.sort(res[c].values)
    holm_ok = (p_sorted < 0.05 / (len(res) - np.arange(len(res)))).cumprod().sum()
    print(f"  {c:12s} {int(holm_ok):2d} / {len(res)}")
