"""Calibration of the twin's class probabilities and concentration ranges, with and without online range
recalibration (rdt/softsensor/recalibration.py), on 2021-2024 (eval_primary) and 2021-2025.

Each variant runs the live replay headless (as `rdt run --no-dashboard`) and uses what the twin recorded:
for every complete lab sample, the class probabilities predicted for its sample time (twin and
persistence reference), and for every lab metal result, the predicted median and 90% range.

Checks per variant and station:
  1. "at least moderately contaminated" (the decision-relevant event): reliability table, Brier
     decomposition, and calibration-in-the-large (mean forecast - observed rate, block-bootstrap 95%);
  2. per class: mean forecast probability vs observed share;
  3. per metal: share of lab results inside the 50/80/90/95% ranges (twin only; the reference records
     no range).
A comparison table of the variants follows.

The recalibration weight (20 lab results) is fixed in advance, not chosen on these results, which would
tune on the evaluation period. 2025 has organic carbon only until August, so later samples there have no
class (they still count for the concentration ranges).

Samples are weekly (Eijsden) or fortnightly (Lobith) and autocorrelated, so the per-bin intervals are too
narrow; the block bootstrap (blocks of ~2 months, as in 04) accounts for that in the overall bias.

Usage: python analysis/06_class_calibration.py [config.yaml]
"""
import dataclasses
import logging
import sys

import numpy as np
import pandas as pd

from rdt.config import load_config
from rdt.monitoring.calibration import (
    block_bootstrap_mean,
    brier_decomposition,
    pit_coverage,
    pit_lognormal,
    reliability_table,
)
from rdt.monitoring.classes import score_class_comparisons
from rdt.quality import QUALITY_CLASSES
from rdt.runtime import build_system

logging.basicConfig(level=logging.WARNING)
pd.set_option("display.width", 160)
BLOCK = {"eijsden": 8, "lobith": 4}          # ~2 months of samples, as in 04
AT_LEAST_MODERATE = QUALITY_CLASSES[2:]
LEVELS = (0.5, 0.8, 0.9, 0.95)
PRIOR_N = 20
VARIANTS = {                                 # name: (replay_end, recalibrate_prior_n)
    "2021-2024 fixed": ("2025-01-01", None),
    "2021-2024 recalibrated": ("2025-01-01", PRIOR_N),
    "2021-2025 fixed": ("2026-01-01", None),
    "2021-2025 recalibrated": ("2026-01-01", PRIOR_N),
}


def event_forecasts(comparisons, reference: bool) -> tuple[np.ndarray, np.ndarray]:
    p = np.array([sum((c.reference_probabilities if reference else c.probabilities)[k] for k in AT_LEAST_MODERATE)
                  for c in comparisons])
    o = np.array([c.observed in AT_LEAST_MODERATE for c in comparisons])
    return p, o


def coverage_table(store, station: str) -> pd.DataFrame:
    lab = pd.DataFrame([vars(c) for c in store.comparisons(station) if not c.censored])
    rows = []
    for metal, g in lab.groupby("metal"):
        cov = pit_coverage(pit_lognormal(g.observed, g.predicted_median, g.upper), LEVELS)
        rows.append({"metal": metal, "n": len(g), **{f"{lv:.0%} range": v for lv, v in cov.items()}})
    return pd.DataFrame(rows)


def report(store, station: str) -> dict:
    """Print the detailed checks for one station; return the numbers for the comparison table."""
    cmp = sorted(store.class_comparisons(station), key=lambda c: c.time_utc)
    print(f"\n--- {station}: {len(cmp)} complete lab samples, "
          f"{np.mean([c.observed in AT_LEAST_MODERATE for c in cmp]):.0%} at least moderately contaminated ---")

    print("\n1. P(at least moderately contaminated)")
    summary = {}
    for name, ref in (("twin", False), ("persistence", True)):
        p, o = event_forecasts(cmp, ref)
        d = brier_decomposition(p, o)
        lo, hi = block_bootstrap_mean(p - o, BLOCK[station])
        table = reliability_table(p, o)
        print(f"\n  {name}: Brier {d['brier']:.3f} = reliability {d['reliability']:.3f} - resolution "
              f"{d['resolution']:.3f} + uncertainty {d['uncertainty']:.3f} (binned, approximate)")
        print(f"  mean forecast - observed rate: {np.mean(p - o):+.3f} (95% block bootstrap {lo:+.3f} to {hi:+.3f})")
        print(table.round(3).to_string(index=False))
        if name == "twin":
            top = table.iloc[-1]
            summary = {"samples": len(cmp), "class Brier": score_class_comparisons(cmp)["brier"],
                       "event Brier": d["brier"], "reliability": d["reliability"], "resolution": d["resolution"],
                       "bias": float(np.mean(p - o)), "top bin forecast": top.mean_forecast,
                       "top bin observed": top.observed}

    print("\n2. Per class: mean forecast probability vs observed share")
    rows = [{"class": k, "twin": np.mean([c.probabilities[k] for c in cmp]),
             "persistence": np.mean([c.reference_probabilities[k] for c in cmp]),
             "observed": np.mean([c.observed == k for c in cmp])} for k in QUALITY_CLASSES]
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\n3. Twin concentration ranges: share of lab results inside each central range (PIT)")
    cov = coverage_table(store, station)
    print(cov.round(3).to_string(index=False))
    gaps = cov[[f"{lv:.0%} range" for lv in LEVELS]].to_numpy() - np.array(LEVELS)
    summary["range error (pp)"] = float(np.abs(gaps).mean() * 100)
    summary["worst 90% range"] = float(cov["90% range"].min())
    return summary


base = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
results = []
for name, (end, prior_n) in VARIANTS.items():
    cfg = dataclasses.replace(
        base,
        periods=dataclasses.replace(base.periods, replay_end=pd.Timestamp(end, tz="UTC")),
        softsensor=dataclasses.replace(base.softsensor, recalibrate_prior_n=prior_n),
    )
    system = build_system(cfg, speed=None)
    system.replayer.run()
    print(f"\n=================== {name} ===================")
    for station in BLOCK:
        results.append({"variant": name, "station": station, **report(system.store, station)})

print("\n=================== Comparison (twin) ===================")
print("class Brier: 4 classes; event: P(at least moderately contaminated); reliability: miscalibration part of")
print("the event Brier (lower is better); resolution: discrimination (higher is better); bias: mean forecast -")
print("observed rate; top bin: forecasts of 0.8-1.0; range error: mean |coverage - nominal| over the 50/80/90/95%")
print("ranges and all metals, in percentage points.\n")
print(pd.DataFrame(results).sort_values(["station", "variant"]).round(3).to_string(index=False))
