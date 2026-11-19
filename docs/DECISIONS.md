# Design decisions

Short record of choices made while porting features from the MVP (branch `port-mvp`), and why. Each entry
says where it applies, so it can be revisited or reversed.

## Quality classes

- Organic matter = 1.724 x organic carbon (`quality.om_per_corg`). The lab reports carbon; the soil-type
  correction needs organic matter. 1.724 is the conventional factor; 2.0 changes class shares by about
  1 percentage point (MVP report).
- Lutum and organic matter are treated as exact. In reality they come from the last lab sample, which can
  be weeks old, so class probabilities are slightly overconfident.
- Missing lutum or organic carbon gives "no class" instead of assuming standard soil
  (`quality/classify.py` raises; the dashboard says so). Assuming standard soil would silently give a
  wrong class.
- The overall class of a sample is the worst class of its five metals, computed per joint draw, so that
  metals that rise and fall together are handled correctly.

## Class evaluation

- Predicted class probabilities are scored against each complete lab sample (all five metals, lutum and
  organic carbon) using that sample's own lutum and organic matter (`monitoring/classes.py`). This
  measures how well the metals are predicted, not the soil properties, and matches the MVP's V4 check.
- The baseline is Sary's persistence reference (last lab value) instead of the MVP's training class
  frequencies: it already runs alongside the twin and is the stronger comparison.

## Calibration

- Calibration is checked on the event "at least moderately contaminated" (the decision-relevant one),
  because four classes with about 300 samples is too thin (`analysis/06_class_calibration.py`).
- Online range recalibration (`softsensor/recalibration.py`) re-estimates each metal's range factor from
  the lab results received so far, with the same 90%-coverage rule Sary uses on training. Its prior weight
  (20 lab results) was fixed in advance, not tuned on the evaluation period.
- Recalibration is off by default (`softsensor.recalibrate_prior_n: null`). It makes the concentration
  ranges honest (coverage error 4.5 -> 2.4-3.0 percentage points) but does not improve the class
  probabilities.
- The class overconfidence comes from the predictions being too extreme, not too narrow: when the twin
  predicts low, lab results come in higher, and vice versa. Fixing that needs a change to the predicted
  values themselves (not yet done).

## Drift detection

- CUSUM (ported from the MVP) is the default drift detector (`monitoring.drift: cusum`); Sary's
  interval-exceedance detector remains available. CUSUM also detects small persistent bias.
- CUSUM settings are the textbook ones (0.5 sd allowance, alarm at 5 sd), not tuned on 2021-2024. They
  give about one alarm a month in 2021-2024 (27 of 43 in 2022), which may be too many for users.

## Representation model

- New parts of the representation model are optional components (`twin/components.py`), switched on by
  name in `twin.components`. They only add to Sary's twin: removing one from the config switches it off.
- Components write the current state of their entities to the store as `EntityState` records (latest
  only, no history) and events as Sary's `Alert` records, so the dashboard keeps reading only the store.
