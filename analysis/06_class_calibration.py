"""Calibration of the twin's class probabilities and concentration ranges on eval_primary (2021-2024).

Runs the live replay headless (as `rdt run --no-dashboard`) and uses what the twin recorded: for every
complete lab sample, the class probabilities predicted for its sample time (twin and persistence
reference), and for every lab metal result, the predicted median and 90% range.

Checks:
  1. "at least moderately contaminated" (the decision-relevant event): reliability table, Brier
     decomposition, and calibration-in-the-large (mean forecast - observed rate, block-bootstrap 95%);
  2. per class: mean forecast probability vs observed share;
  3. per metal: PIT coverage of the 50/80/90/95% ranges (twin only; the reference records no range).

Samples are weekly (Eijsden) or fortnightly (Lobith) and autocorrelated, so the per-bin intervals are too
narrow; the block bootstrap (blocks of ~2 months, as in 04) accounts for that in the overall bias.

Usage: python analysis/06_class_calibration.py [config.yaml]
"""
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
from rdt.quality import QUALITY_CLASSES
from rdt.runtime import build_system

logging.basicConfig(level=logging.WARNING)
pd.set_option("display.width", 140)
BLOCK = {"eijsden": 8, "lobith": 4}          # ~2 months of samples, as in 04
AT_LEAST_MODERATE = QUALITY_CLASSES[2:]

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
system = build_system(cfg, speed=None)
system.replayer.run()


def event_forecasts(comparisons, reference: bool) -> tuple[np.ndarray, np.ndarray]:
    p = np.array([sum((c.reference_probabilities if reference else c.probabilities)[k] for k in AT_LEAST_MODERATE)
                  for c in comparisons])
    o = np.array([c.observed in AT_LEAST_MODERATE for c in comparisons])
    return p, o


for station in BLOCK:
    cmp = sorted(system.store.class_comparisons(station), key=lambda c: c.time_utc)
    print(f"\n=== {station}: {len(cmp)} complete lab samples, "
          f"{np.mean([c.observed in AT_LEAST_MODERATE for c in cmp]):.0%} at least moderately contaminated ===")

    print("\n1. P(at least moderately contaminated)")
    for name, ref in (("twin", False), ("persistence", True)):
        p, o = event_forecasts(cmp, ref)
        d = brier_decomposition(p, o)
        lo, hi = block_bootstrap_mean(p - o, BLOCK[station])
        print(f"\n  {name}: Brier {d['brier']:.3f} = reliability {d['reliability']:.3f} - resolution "
              f"{d['resolution']:.3f} + uncertainty {d['uncertainty']:.3f} (binned, approximate)")
        print(f"  mean forecast - observed rate: {np.mean(p - o):+.3f} (95% block bootstrap {lo:+.3f} to {hi:+.3f})")
        print(reliability_table(p, o).round(3).to_string(index=False))

    print("\n2. Per class: mean forecast probability vs observed share")
    rows = []
    for k in QUALITY_CLASSES:
        rows.append({"class": k, "twin": np.mean([c.probabilities[k] for c in cmp]),
                     "persistence": np.mean([c.reference_probabilities[k] for c in cmp]),
                     "observed": np.mean([c.observed == k for c in cmp])})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\n3. Twin concentration ranges: share of lab results inside each central range (PIT)")
    lab = pd.DataFrame([vars(c) for c in system.store.comparisons(station) if not c.censored])
    rows = []
    for metal, g in lab.groupby("metal"):
        cov = pit_coverage(pit_lognormal(g.observed, g.predicted_median, g.upper))
        rows.append({"metal": metal, "n": len(g), **{f"{lv:.0%} range": v for lv, v in cov.items()}})
    print(pd.DataFrame(rows).round(3).to_string(index=False))
