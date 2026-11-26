"""Comparison of the options for improving the twin's calibration, on 2021-2024 and 2021-2025.

The calibration check (06_class_calibration.py) found the class probabilities overconfident at the
extremes, and traced it to the predicted values being too extreme rather than the ranges too narrow.
Part 1 locates that: the error of each lab prediction is regressed on the two parts of the Kalman
prediction, the sensor part (B x: discharge, turbidity, season) and the level's deviation from its own
average over the previous ~26 results. A negative slope means that part overshoots.

Part 2 compares the options, each without tuning on the evaluation period:
  baseline                 Sary's Kalman soft sensor, without range recalibration
  range recalibration      ranges re-estimated from the lab results so far (softsensor/recalibration.py)
  one-step level noise     the level drift Q rescaled to minimise training one-step-ahead error
                           (softsensor/kalman.py, noise_fit="one_step"): fixes the cause, on training
  level correction         prediction + c * level deviation, with c learned online from the lab results
                           so far (prior weight 20 results towards c = 0); metal predictions only
  + probability recal.     online Platt scaling of P(at least moderately contaminated)
                           (monitoring/probability_recalibration.py, prior weight 20 samples)
A result counts as known from sample time + lab turnaround (softsensor.lab_delay_d), as in the replay.

Usage: python analysis/07_calibration_options.py [config.yaml]
"""
import dataclasses
import logging
import sys

import numpy as np
import pandas as pd

from rdt.config import load_config
from rdt.monitoring.calibration import block_bootstrap_mean, brier_decomposition, pit_coverage, pit_lognormal, reliability_table
from rdt.monitoring.classes import score_class_comparisons
from rdt.monitoring.probability_recalibration import online_platt
from rdt.quality import QUALITY_CLASSES
from rdt.runtime import build_system

logging.basicConfig(level=logging.WARNING)
pd.set_option("display.width", 200)
STATIONS = ("eijsden", "lobith")
AT_LEAST_MODERATE = QUALITY_CLASSES[2:]
LEVELS = (0.5, 0.8, 0.9, 0.95)
PRIOR_N = 20
SLOW_WINDOW = 26                       # results in the level's own running average (~6 months at Eijsden)
PERIODS = {"2021-2024": "2025-01-01", "2021-2025": "2026-01-01"}
# Every option is set explicitly, so the rows do not change with the config's defaults.
REPLAYS = {                            # name: softsensor overrides
    "baseline": {"recalibrate_prior_n": None, "kalman_noise_fit": "variogram"},
    "range recalibration": {"recalibrate_prior_n": PRIOR_N, "kalman_noise_fit": "variogram"},
    "one-step level noise": {"recalibrate_prior_n": None, "kalman_noise_fit": "one_step"},
}


def lab_errors(system, station: str) -> pd.DataFrame:
    """Per uncensored lab result: log error, prediction, its sensor part and level part, 90% upper bound."""
    kalman = getattr(system.soft_sensor, "inner", system.soft_sensor)
    params, x_at = kalman._params[station], {g.t: g.x for g in kalman._groups[station]}
    rows = []
    for c in system.store.comparisons(station):
        if c.censored:
            continue
        pred = np.log(c.predicted_median)
        sensor = float(params.B[kalman.metals.index(c.metal)] @ x_at[pd.Timestamp(c.time_utc)])
        rows.append({"t": pd.Timestamp(c.time_utc), "metal": c.metal, "observed": c.observed,
                     "median": c.predicted_median, "upper": c.upper, "e": np.log(c.observed) - pred,
                     "sensor": sensor, "level": pred - sensor})
    df = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
    df["level_dev"] = df["level"] - df.groupby("metal")["level"].transform(
        lambda s: s.shift(1).rolling(SLOW_WINDOW, min_periods=8).mean())
    return df


def overshoot_slopes(df: pd.DataFrame) -> dict:
    """Slopes of the error on the sensor part and on the level deviation (metal means removed), with a
    95% bootstrap interval over quarters."""
    d = df.dropna(subset=["level_dev"])
    cols = ["e", "sensor", "level_dev"]
    d = d[cols] - d.groupby(d["metal"])[cols].transform("mean")
    X, y = d[["sensor", "level_dev"]].to_numpy(), d["e"].to_numpy()
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    quarter = df.loc[d.index, "t"].dt.tz_localize(None).dt.to_period("Q").astype(str).to_numpy()
    blocks, rng, boot = np.unique(quarter), np.random.default_rng(0), []
    for _ in range(1000):
        idx = np.concatenate([np.flatnonzero(quarter == q) for q in rng.choice(blocks, len(blocks))])
        boot.append(np.linalg.lstsq(X[idx], y[idx], rcond=None)[0])
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    explained = 1 - np.var(y - X[:, 1] * beta[1]) / np.var(y)
    return {"sensor slope": f"{beta[0]:+.2f} [{lo[0]:+.2f}, {hi[0]:+.2f}]",
            "level slope": f"{beta[1]:+.2f} [{lo[1]:+.2f}, {hi[1]:+.2f}]",
            "error variance explained by level overshoot": f"{explained:.1%}"}


def level_corrected_errors(df: pd.DataFrame, delay: pd.Timedelta) -> np.ndarray:
    """Errors after adding c * level deviation, c per metal from the results available before each sample."""
    out = df["e"].to_numpy().copy()
    for _, g in df.groupby("metal"):
        idx, avail = g.index.to_numpy(), (g["t"] + delay).to_numpy()
        dev, e, t = g["level_dev"].to_numpy(), g["e"].to_numpy(), g["t"].to_numpy()
        for k in range(len(g)):
            past = (avail < t[k]) & np.isfinite(dev)
            if not np.isfinite(dev[k]) or not past.any():
                continue
            dd, ee = dev[past], e[past]
            c = (dd @ ee) / (dd @ dd + PRIOR_N * np.mean(dd**2))
            out[idx[k]] = e[k] - c * dev[k]
    return out


def range_error(df: pd.DataFrame, median_factor: np.ndarray | None = None) -> float:
    """Mean |coverage - nominal| of the 50/80/90/95% ranges over metals, in percentage points."""
    gaps = []
    for _, g in df.groupby("metal"):
        median = g["median"] * (1 if median_factor is None else median_factor[g.index])
        cov = pit_coverage(pit_lognormal(g["observed"], median, g["upper"] * median / g["median"]), LEVELS)
        gaps += [abs(cov[lv] - lv) for lv in LEVELS]
    return 100 * float(np.mean(gaps))


def event_scores(p: np.ndarray, o: np.ndarray) -> dict:
    d, top = brier_decomposition(p, o), reliability_table(p, o).iloc[-1]
    bottom = reliability_table(p, o).iloc[0]
    return {"event Brier": d["brier"], "reliability": d["reliability"], "resolution": d["resolution"],
            "bias": float(np.mean(p - o)),
            "top bin (forecast / observed)": f"{top.mean_forecast:.2f} / {top.observed:.2f}",
            "bottom bin (forecast / observed)": f"{bottom.mean_forecast:.2f} / {bottom.observed:.2f}"}


base = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
delay = pd.Timedelta(days=base.softsensor.lab_delay_d)
rows, slopes = [], []
for period, end in PERIODS.items():
    for name, overrides in REPLAYS.items():
        cfg = dataclasses.replace(
            base,
            periods=dataclasses.replace(base.periods, replay_end=pd.Timestamp(end, tz="UTC")),
            softsensor=dataclasses.replace(base.softsensor, **overrides),
        )
        system = build_system(cfg, speed=None)
        system.replayer.run()
        for station in STATIONS:
            lab = lab_errors(system, station)
            cmp = sorted(system.store.class_comparisons(station), key=lambda c: c.time_utc)
            p = np.array([sum(c.probabilities[k] for k in AT_LEAST_MODERATE) for c in cmp])
            o = np.array([c.observed in AT_LEAST_MODERATE for c in cmp])
            t = np.array([pd.Timestamp(c.time_utc) for c in cmp])
            common = {"period": period, "station": station, "samples": len(cmp)}
            rows.append({**common, "option": name, "median |log error|": lab["e"].abs().median(),
                         "range error (pp)": range_error(lab), "class Brier": score_class_comparisons(cmp)["brier"],
                         **event_scores(p, o)})
            if name in ("baseline", "range recalibration"):
                p_new = online_platt(p, o, t, t + delay, prior_n=PRIOR_N)
                lo, hi = block_bootstrap_mean((p_new - o) ** 2 - (p - o) ** 2, block=8 if station == "eijsden" else 4)
                rows.append({**common, "option": f"{name} + probability recal.",
                             "median |log error|": lab["e"].abs().median(), "range error (pp)": range_error(lab),
                             **event_scores(p_new, o), "event Brier change 95%": f"{lo:+.3f} to {hi:+.3f}"})
            if name == "baseline":
                slopes.append({"period": period, "station": station, **overshoot_slopes(lab)})
                e_new = level_corrected_errors(lab, delay)
                rows.append({**common, "option": "level correction", "median |log error|": np.median(np.abs(e_new)),
                             "range error (pp)": range_error(lab, np.exp(lab["e"].to_numpy() - e_new))})

print("\n=================== 1. Which part of the prediction overshoots (baseline) ===================")
print("Slope of the log error on each part of the prediction; negative = that part overshoots.\n")
print(pd.DataFrame(slopes).to_string(index=False))

print("\n=================== 2. Options ===================")
print("median |log error|: metal predictions (0.10 = about 10%); range error: mean |coverage - nominal| of the")
print("50/80/90/95% ranges; class Brier: 4 classes; event: P(at least moderately contaminated), with reliability")
print("(miscalibration, lower is better) and resolution (discrimination, higher is better); top / bottom bin:")
print("forecasts of 0.8-1.0 / 0.0-0.2. The probability recalibration adjusts the event only, so it has no class Brier.")
print("Event Brier change: probability recal. minus without, 95% block bootstrap.\n")
table = pd.DataFrame(rows).sort_values(["period", "station"], kind="stable")
print(table.round(3).to_string(index=False))
