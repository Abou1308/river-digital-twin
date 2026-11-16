"""Evaluation of the soft sensors (package `rdt.softsensor`) on eval_primary 2021-2024.

Setup, as in the live twin:
  * fitted on training lab samples (2012-2020) with inputs from cleaned readings (training-only QC rules);
  * test inputs from readings as delivered, with only the automatic arrival rules;
  * a lab result becomes available `delay` days after its sample time; the prediction for each test
    sample uses only results that had arrived strictly before that sample's time (no peeking);
  * 2025 (eval_secondary) is not used.

Models: kalman (main), ridge_log (sensors + last known lab value; best model in 03), persistence.
Per station / metal / delay: median absolute % error, coverage of the 90% prediction interval
(should be ~90%), and Diebold-Mariano (HAC) + block-bootstrap tests of kalman vs the others.

Usage: python analysis/04_soft_sensor_evaluation.py [config.yaml]
"""
import sys
import warnings

import numpy as np
import pandas as pd

from rdt.config import load_config
from rdt.data import load_model_table, load_readings
from rdt.features import max_window
from rdt.monitoring import block_bootstrap_median_diff, diebold_mariano, holm_count
from rdt.softsensor import build_table, make_soft_sensor

warnings.filterwarnings("ignore")
cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
METALS, FEATS = cfg.softsensor.metals, cfg.softsensor.features
DELAYS = cfg.softsensor.eval_lab_delays_d
MODELS = ["kalman", "ridge_log", "persistence"]
BLOCK = {"eijsden": 8, "lobith": 4}          # ~2 months of samples
cut = cfg.periods.train_end
Z90 = 1.6448536269514722

mt = load_model_table(cfg.data_dir)
mt = mt[mt.station.isin(["eijsden", "lobith"])]
readings = load_readings(cfg.data_dir, end=cfg.periods.replay_end)
train = build_table(mt[mt.period == "training"], readings[readings.time_utc < cut], METALS, FEATS, training=True)
test = build_table(mt[mt.period == "eval_primary"],
                   readings[readings.time_utc >= cut - max_window(FEATS) - pd.Timedelta(days=1)], METALS, FEATS, training=False)
print(f"training samples: {len(train)}, test samples (eval_primary): {len(test)}")
print(f"features: {', '.join(FEATS)}")
print("share of test samples with a missing input:",
      test[list(FEATS)].isna().any(axis=1).groupby(test.station).mean().round(3).to_dict(), "\n")

rows = []
for delay in DELAYS:
    for name in MODELS:
        s = make_soft_sensor(name, METALS, FEATS, lab_delay_d=delay)
        s.fit(train, cut)
        for _, r in test.iterrows():
            f = {k: r[k] for k in FEATS}
            for m in METALS:
                if pd.notna(r[m]):
                    s.assimilate(r.station, r.sample_time_utc, m, r[m], False, features=f,
                                 available_utc=r.sample_time_utc + pd.Timedelta(days=delay))
        for _, r in test.iterrows():
            est = s.predict(r.station, r.sample_time_utc, {k: r[k] for k in FEATS}, as_of=r.sample_time_utc)
            sd = est.sd_log()
            for i, m in enumerate(METALS):
                if pd.notna(r[m]):
                    rows.append(dict(delay_d=delay, model=name, station=r.station, metal=m, t=r.sample_time_utc,
                                     obs=r[m], mu=est.mean_log[i], sd=sd[m]))
        if name == "kalman" and delay == DELAYS[0]:
            kal = s

P = pd.DataFrame(rows)
P["ape"] = np.abs(np.exp(P.mu) - P.obs) / P.obs * 100
P["abs_log_err"] = np.abs(P.mu - np.log(P.obs))
P["in90"] = np.abs(np.log(P.obs) - P.mu) <= Z90 * P.sd
pd.set_option("display.width", 220)

print("=== Median absolute % error (rows station/metal; columns model x delay in days)")
print(P.pivot_table(index=["station", "metal"], columns=["model", "delay_d"], values="ape", aggfunc="median")
      .reindex(columns=MODELS, level=0).round(1).to_string())

print("\n=== Coverage of the 90% prediction interval, % (target 90)")
print((P.pivot_table(index=["station", "metal"], columns=["model", "delay_d"], values="in90", aggfunc="mean") * 100)
      .reindex(columns=MODELS, level=0).round(0).to_string())

print("\n=== kalman vs other models: diff = other median % error - kalman median % error (positive = kalman "
      "better), 95% block-bootstrap interval, DM test (HAC) on |log error|")
res = []
for (delay, st, m), g in P.groupby(["delay_d", "station", "metal"]):
    w = g.pivot_table(index="t", columns="model", values=["ape", "abs_log_err"]).sort_index()
    for other in ["ridge_log", "persistence"]:
        stat, p = diebold_mariano(w["abs_log_err"][other] - w["abs_log_err"]["kalman"], lag=BLOCK[st] - 1)
        lo, hi = block_bootstrap_median_diff(w["ape"][other].to_numpy(), w["ape"]["kalman"].to_numpy(), BLOCK[st])
        diff = w["ape"][other].median() - w["ape"]["kalman"].median()
        verdict = ("kalman better" if p < 0.05 and lo > 0 else "kalman worse" if p < 0.05 and hi < 0
                   else "DM only" if p < 0.05 else "no sig. difference")
        res.append(dict(vs=other, delay_d=delay, station=st, metal=m, diff=diff, boot_lo=lo, boot_hi=hi,
                        dm_stat=stat, dm_p=p, verdict=verdict))
R = pd.DataFrame(res)
out = R.copy()
for c in ["diff", "boot_lo", "boot_hi", "dm_stat"]:
    out[c] = out[c].round(1)
out["dm_p"] = out.dm_p.map(lambda v: f"{v:.1e}" if v < 1e-3 else f"{v:.3f}")
for other in ["ridge_log", "persistence"]:
    print(f"\n--- vs {other}")
    print(out[out.vs == other].drop(columns="vs").to_string(index=False))
print("\nSummary (count of 10 station/metal combinations) and Holm-corrected DM rejections:")
for (vs, delay), g in R.groupby(["vs", "delay_d"]):
    v = g.verdict.value_counts().to_dict()
    print(f"  vs {vs:11s} delay {delay:2d} d: {v}  | Holm: {holm_count(g.dm_p.values)}/10")

print("\n=== Kalman model, fitted on training (diagnostics)")
for st, d in kal.diagnostics.items():
    print(f"\n--- {st}")
    print("B (effect of +1 sd of each input on ln metal):")
    print(d["B"].round(3).to_string())
    q, r = np.sqrt(np.diag(d["Q"].to_numpy())), np.sqrt(np.diag(d["R"].to_numpy()))
    print("level drift sd over 1 year (ln units): " + ", ".join(f"{m} {v:.2f}" for m, v in zip(METALS, q * np.sqrt(365))))
    print("sample-level noise sd (ln units):     " + ", ".join(f"{m} {v:.2f}" for m, v in zip(METALS, r)))
    z = d["train_innovation_z"]
    print("training innovations: sd of z (target ~1): " + ", ".join(f"{m} {z[m].std():.2f}" for m in METALS))
    print("interval widening factor (from training): " + ", ".join(f"{m} {v:.2f}" for m, v in d["interval_scale"].items()))
    print("variogram (binned robust variance of residual differences vs straight-line fit 2R + Q*lag):")
    vg = pd.concat({m: d["variogram"][m].set_index("lag_d")[["gamma", "fit"]] for m in METALS}, axis=1)
    print(vg.round(3).to_string())
