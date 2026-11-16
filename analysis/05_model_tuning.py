"""Can extra sensor inputs improve the Kalman soft sensor? Tested on a validation split *inside* the
training period, so the 2021-2024 test is not used for model choices.

    fit on        training lab samples before 2018-01-01 (inputs from cleaned readings)
    validate on   training lab samples 2018-2020 (inputs from readings as delivered + arrival rules,
                  lab results available after `lab_delay_d` days, as in the live twin)

Input sets for the Kalman model (see rdt/features/windows.py):
    base     current config features
    +long    30-day means of discharge and SPM (antecedent conditions, slow wash-out)
    +peaks   30-day max discharge, 7-day max turbidity (recent flood / resuspension)
    +3d      3-day means of discharge and turbidity (shorter event memory)
    +all     all of the above
Reference: ridge_log and persistence on the same split.

Note: with ln-transformed inputs, ratios such as SPM/turbidity (grain-size proxy) or 24h/7d discharge
(rising vs falling limb) are already linear combinations of the base inputs, so they add nothing new.

Outcome (analysis/results/05_model_tuning.txt): differences are within noise, so the model keeps the
base inputs. During development, estimating Q and R by maximum likelihood and estimating B jointly
with the level were also tried on this split; both scored worse and their code was removed.

Usage: python analysis/05_model_tuning.py [config.yaml]
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd

from rdt.config import load_config
from rdt.data import load_model_table, load_readings
from rdt.features import max_window
from rdt.softsensor import build_table, make_soft_sensor

warnings.filterwarnings("ignore")
cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
METALS = cfg.softsensor.metals
BASE = tuple(cfg.softsensor.features)
DELAY = cfg.softsensor.lab_delay_d
VAL_START = pd.Timestamp("2018-01-01", tz="UTC")
VAL_END = cfg.periods.train_end
Z90 = 1.6448536269514722

FEATURE_SETS = {
    "base": BASE,
    "+long": BASE + ("q_mean_30d", "spm_mean_30d"),
    "+peaks": BASE + ("q_max_30d", "turb_max_7d"),
    "+3d": BASE + ("q_mean_3d", "turb_mean_3d"),
    "+all": BASE + ("q_mean_30d", "spm_mean_30d", "q_max_30d", "turb_max_7d", "q_mean_3d", "turb_mean_3d"),
}
CANDIDATES = [("kalman", fs) for fs in FEATURE_SETS] + [("ridge_log", "base"), ("persistence", "base")]
ALL_FEATS = tuple(dict.fromkeys(sum(FEATURE_SETS.values(), ())))

mt = load_model_table(cfg.data_dir)
mt = mt[mt.station.isin(["eijsden", "lobith"]) & (mt.period == "training")]
readings = load_readings(cfg.data_dir, end=VAL_END)
train = build_table(mt[mt.sample_time_utc < VAL_START], readings[readings.time_utc < VAL_START],
                    METALS, ALL_FEATS, training=True)
val = build_table(mt[mt.sample_time_utc >= VAL_START],
                  readings[readings.time_utc >= VAL_START - max_window(ALL_FEATS) - pd.Timedelta(days=1)],
                  METALS, ALL_FEATS, training=False)
print(f"fit: {len(train)} samples before {VAL_START.date()}, validate: {len(val)} samples "
      f"{VAL_START.date()}..{VAL_END.date()}, lab delay {DELAY:g} d\n")

rows = []
for kind, fs in CANDIDATES:
    feats = FEATURE_SETS[fs]
    t0 = time.time()
    s = make_soft_sensor(kind, METALS, feats, lab_delay_d=DELAY)
    s.fit(train, VAL_START)
    for _, r in val.iterrows():
        for m in METALS:
            if pd.notna(r[m]):
                s.assimilate(r.station, r.sample_time_utc, m, r[m], False, features={k: r[k] for k in feats},
                             available_utc=r.sample_time_utc + pd.Timedelta(days=DELAY))
    for _, r in val.iterrows():
        est = s.predict(r.station, r.sample_time_utc, {k: r[k] for k in feats}, as_of=r.sample_time_utc)
        sd = est.sd_log()
        for i, m in enumerate(METALS):
            if pd.notna(r[m]):
                rows.append(dict(model=f"{kind} | {fs}", station=r.station, metal=m, obs=r[m],
                                 mu=est.mean_log[i], sd=sd[m]))
    print(f"  {kind:12s} {fs:7s} done in {time.time() - t0:5.1f} s", flush=True)

P = pd.DataFrame(rows)
P["ape"] = np.abs(np.exp(P.mu) - P.obs) / P.obs * 100
P["abs_log"] = np.abs(P.mu - np.log(P.obs))
P["in90"] = np.abs(np.log(P.obs) - P.mu) <= Z90 * P.sd
order = list(dict.fromkeys(P.model))
pd.set_option("display.width", 250)

print("\n=== Median absolute % error on validation (2018-2020)")
tab = P.pivot_table(index="model", columns=["station", "metal"], values="ape", aggfunc="median").reindex(order)
tab["mean"] = tab.mean(axis=1)
print(tab.round(1).to_string())
print("\n=== Summary per model (averaged over the 10 station/metal combinations)")
summ = P.groupby(["model", "station", "metal"]).agg(ape=("ape", "median"), abs_log=("abs_log", "mean"),
                                                    cov=("in90", "mean")).groupby("model").mean().reindex(order)
summ["cov"] *= 100
summ.columns = ["median % error", "mean |log error|", "90% coverage %"]
print(summ.round(3).to_string())

