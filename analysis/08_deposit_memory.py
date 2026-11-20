"""Sensitivity of the fresh deposit (representation/deposit.py) to its memory, on 2021-2024.

No source gives the residence time of fresh deposit (docs/DECISIONS.md); the MVP used 30 days. For
memories of 10-90 days, each a full replay, this reports per reach:
  1. how often the deposit's reported class changes, and the share of time in each class;
  2. how closely the deposit's median content matches the downstream lab samples (Keizersveer, Kampen),
     which are the nearest observation of what settles at the end of each reach. The per-metal level
     difference (tributaries, sorting) is removed with the median log ratio over these same samples, so
     this measures how well the deposit follows the downstream variation, not its level. For
     reference: the error of the last downstream lab value (persistence).

Usage: python analysis/08_deposit_memory.py [config.yaml]
"""
import copy
import dataclasses
import logging
import sys
from collections import Counter

import numpy as np
import pandas as pd

from rdt.config import load_config
from rdt.quality import QUALITY_CLASSES
from rdt.runtime import build_system

logging.basicConfig(level=logging.WARNING)
pd.set_option("display.width", 200)
MEMORIES = (10, 14, 30, 60, 90)


def run(cfg, residence_days: float) -> tuple[list, list, object]:
    """Replay with the given memory; return the reach entity states over time, the class alerts and the system."""
    representation = copy.deepcopy(cfg.representation)
    representation["reach"]["residence_days"] = residence_days
    system = build_system(dataclasses.replace(cfg, representation=representation), speed=None)
    history, record = [], system.store.record_entity

    def keep(state):
        if state.kind == "reach":
            history.append(state)
        record(state)

    system.store.record_entity = keep
    system.replayer.run()
    alerts = [a for a in system.store.alerts(limit=10_000) if a.kind == "class_change"]
    return history, alerts, system


def downstream_errors(history: list, system, reach) -> tuple[float, float, int]:
    """Median |log error| of the deposit median and of persistence against downstream lab results."""
    states = [s for s in history if s.id == reach.id and "deposit_median" in s.values]
    times = pd.DatetimeIndex([s.time_utc for s in states])
    rows = []
    for metal in system.cfg.softsensor.metals:
        lab = [m for m in system.store.readings(f"{reach.downstream}_spm_{metal}") if not m.censored]
        for m in lab:
            i = times.searchsorted(pd.Timestamp(m.time_utc), side="right") - 1
            known = [p for p in lab if p.arrival_utc <= m.time_utc]      # results that had arrived by then
            if i < 0 or not known:
                continue
            rows.append({"metal": metal, "ratio": np.log(m.value / states[i].values["deposit_median"][metal]),
                         "persistence": abs(np.log(m.value / known[-1].value))})
    d = pd.DataFrame(rows)
    spread = (d["ratio"] - d.groupby("metal")["ratio"].transform("median")).abs()
    return float(spread.median()), float(d["persistence"].median()), len(d)


base = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
rows = []
for days in MEMORIES:
    history, alerts, system = run(base, days)
    for reach in system.network.reaches:
        classes = Counter(s.values["class"] for s in history if s.id == reach.id and "class" in s.values)
        total = sum(classes.values())
        err, persistence, n = downstream_errors(history, system, reach)
        rows.append({"reach": reach.id, "memory (days)": days,
                     "class changes": sum(a.message.startswith(reach.id) for a in alerts),
                     **{f"time {c.split()[0]}": classes.get(c, 0) / total for c in QUALITY_CLASSES},
                     "downstream |log err|": err, "persistence |log err|": persistence, "downstream n": n})

print("Deposit memory sensitivity, 2021-2024. time <class>: share of 6-h steps with that reported class.")
print("downstream |log err|: deposit median vs downstream lab, level removed per metal (see docstring).\n")
print(pd.DataFrame(rows).sort_values(["reach", "memory (days)"]).round(3).to_string(index=False))
