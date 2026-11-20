# Design decisions

Short record of choices made while porting features from the MVP (branch `port-mvp`), and why. Each entry
says where it applies, so it can be revisited or reversed.

## Quality classes

- Organic matter = 1.97 x organic carbon (`quality.om_per_corg`). The lab reports carbon; the soil-type
  correction needs organic matter. 1.97 was measured for Dutch freshwater sediments (see the input
  assumptions below); the conventional 1.724 (used in the MVP) has no clear source for sediments. The
  regulation expects organic matter measured by loss on ignition, which the data does not include.
  Switching between the two changes class shares by about 1 percentage point (MVP report). On
  2021-2024 the Eijsden class Brier goes from 0.445 to 0.470 (persistence 0.599 to 0.635), because more
  samples then sit near a class boundary. The factor also changes the observed classes, so scores under
  the two factors are not comparable and the choice is made on the source, not on the score.
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
- Recalibration is on by default (`softsensor.recalibrate_prior_n: 20`; null switches it off). It
  halves the range error (4.1-4.5 -> 2.2-3.0 percentage points) at both stations in both periods, and
  leaves the class probabilities unchanged.
- The class overconfidence comes from the predicted values, not the ranges. Of the two parts of a
  Kalman prediction, the level overshoots: after it moves, lab results come back part of the way
  (slope -0.23 at Eijsden, -0.37 at Lobith). The sensor part does not overshoot. The overshoot explains
  only 3-5% of the error variance, so the gain from fixing it is small (analysis/07_calibration_options.py).
- Options compared (all untuned on the evaluation period):
  - Refit how fast the level follows lab results on training (`softsensor.kalman_noise_fit: one_step`):
    worse. Training prefers a level that follows more closely (multiplier about 3), while 2021-2025
    shows it should follow less. Kept as an option for reproducibility; off.
  - Correct each prediction by a learned share of the level's deviation, online: metal errors slightly
    worse at Eijsden, slightly better at Lobith. Not adopted.
  - Online Platt scaling of P(at least moderately contaminated) (`monitoring/probability_recalibration.py`):
    reduces the miscalibration at Eijsden (reliability 0.017 -> 0.006; forecasts of about 0.95 come true
    88% instead of 78%), but the event Brier gain is not significant (95% interval -0.02 to +0.01), it
    makes the mean bias larger, it does nothing at Lobith, and it covers one event, not the four classes
    the dashboard shows. Not used in the live twin; worth rechecking with more data.
- None of the options improves resolution (telling high from low samples apart); that is limited by
  what the sensors say about the metals.

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

## Sensor health

- Each streamed sensor is ok, rejected (last reading failed the arrival rules) or stale
  (`representation/sensor_health.py`). This makes data gaps part of the twin's state instead of
  something users must spot in the charts.
- Stale means no reading for 3 x the nominal interval in sensors.csv, but at least 24 h. 24 h is the
  shortest window the soft sensor uses, so shorter gaps do not change its inputs. With 6 h, the weekly
  ~7-hour maintenance gaps in Lobith turbidity gave about 95 warnings in 2021-2024; with 24 h, 31 remain.
- Gaps are measured in arrival time, so a lab result counts when it arrives, not when it was sampled.
- Turbidity and suspended matter are checked against each other daily (`representation/turbidity_spm.py`,
  ported from the MVP): ln(SPM) = a + b ln(1 + 24-h turbidity), fitted on 2012-2020, with a CUSUM on the
  residuals. This catches a sensor drifting, which the metal checks cannot see.
- The daily residuals are autocorrelated (lag-1 about 0.45), so a textbook CUSUM alarms far too often
  (70 alarms in 2021-2024). The residuals are scaled by sqrt((1 - rho) / (1 + rho)) using the training
  autocorrelation, and alarms are grouped into episodes (one alert, then "consistent" again after 30 quiet
  days). Result: 4 episodes, mainly a lasting shift at Lobith from late 2023 (SPM lower than turbidity
  implies), which the flux component's turbidity fallback should keep in mind.

## Hydrological regime

- Each modelled station is in base flow, rising, peak or recession (`representation/regime.py`, ported
  from the MVP), updated hourly from the trailing 24-h mean discharge and its change over 24 h.
- Thresholds are training percentiles per station (peak at P90, recession only above P50, a trend above
  P75 of 24-h changes), so peak means equally rare flow on the Maas and the Rhine. Entering rising or
  recession needs a clear trend; staying in it only the sign, so small wiggles do not flip the state.
- Only entering and leaving peak raise an alert. All changes gave 743 alerts in 2021-2024 (525 at
  Eijsden, mostly short rising / recession flickers on the regulated Maas), which would bury the
  warnings; peak only gives 115. The current regime of each station is always in its entity.
- Regime changes are stored as `Alert` records (kind "regime_transition", severity info), since Sary's
  store has no separate event log. A separate event log would change the store interface; not done.

## Reaches and fresh deposit

- Each reach (`representation/reach.py`) gets, every 6 h, the sediment load from its upstream station:
  share x 24-h mean discharge x SPM. The reach settings (travel time, share table) live in the
  component's config (`representation.reach.reaches`), not in Sary's `Reach`, so the component can be
  removed without touching the network.
- Rhine split: the IJssel share is interpolated on Lobith discharge in the researched table (below);
  suspended sediment is assumed to divide like the water. The Maas reach takes all of Eijsden's flow;
  its tributaries are not modelled (the downstream comparison is meant to learn the dilution).
- Travel time is fixed per reach at its mean-flow value (Maas 7 days, IJssel 2 days). It only delays
  inputs to a deposit with a 30-day memory, so the low-flow travel times of the Maas (3-6 weeks) are
  the main case where this is wrong.
- SPM is the latest daily composite (at most 48 h old), otherwise estimated from turbidity with the
  training relation. In 2021-2024 the turbidity fallback was used for 30% of steps at Eijsden and 4%
  at Lobith. At Lobith that relation overestimates SPM since late 2023 (see Sensor health).
- The fresh deposit (`representation/deposit.py`, ported from the MVP) is the load-weighted average
  metal content with a 30-day exponential memory. Prediction errors are treated as fully correlated in
  time, so averaging does not narrow the deposit's range; this is the conservative choice.
- The deposit's class uses load-weighted averages of lutum and organic carbon over the same memory. With
  the last single lab value, the class jumped with every sample (71 class changes in 2021-2024 even with
  hysteresis); with averages, 19. The averages are rounded to 0.5 percentage points so the corrected
  limits can be reused; that is within lab precision.
- The reported class changes only when another class has probability >= 0.6 (hysteresis). Without it,
  a deposit near a boundary flipped weekly. 0.6 was chosen before looking at the results.
- Memory: 30 days is kept (`analysis/08_deposit_memory.py`, 10-90 days tested). On the Maas it gives
  the closest match to the Keizersveer samples (median error 8.3%, against 12.8% at 10 days and 8.4-8.8%
  at 60-90 days); on the Rhine the match barely depends on it (9.3-10.8%). This check removes the level
  difference per metal using the same samples, so it flatters the deposit against the last-value
  baseline (8.5% at Keizersveer, 14.8% at Kampen).
- The deposit's class is sensitive to the memory: the share of time the Maas deposit is moderately
  contaminated is 47% at 10 days, 27% at 30 days and 13% at 90 days (Rhine: 21%, 14%, 6%). Longer
  memories smooth out the high-content episodes. This is a real uncertainty of the reach classes, not
  something the data can settle.
- The deposit is a proxy, not a prediction of a measured bed sample: there is no transport physics and
  no bed sampling in the data to check it against. The downstream SPM samples (Keizersveer, Kampen) are
  the nearest check.

## Downstream check

- Lab results at Keizersveer and Kampen (quarterly) are compared with the upstream estimate one travel
  time earlier, times a learned factor per reach and metal (`representation/downstream.py`). The factor
  is the median log ratio downstream / upstream over training lab pairs (2012-2020, 37-53 pairs). It
  absorbs tributary dilution and sorting. Maas: Cd 0.77, Cu 0.91, Ni 1.10, Pb 1.20, Zn 0.96;
  IJssel: Cd 1.23, others 0.96-1.03.
- Results are stored as `LabComparison` records for the downstream station, and a result outside the
  95% range raises a "downstream_mismatch" alert (7 in 2021-2024).
- This needs the "optional" sensor tier streamed (`replay.tiers: [core, optional]`); it also streams the
  water samples at Eijsden and Lobith, which nothing uses yet and which raised no data-gap alerts.
- Result on 2021-2024: inside the 90% range 92% (Keizersveer) and 95% (Kampen). At Kampen the upstream
  estimate is closer than the last Kampen lab value (median error 9% vs 15%). At Keizersveer it is not
  (17% vs 9%): Maas sediment at Keizersveer is much steadier than at Eijsden. Averaging Eijsden over
  longer windows helps (90 days: about 10%), which suggests the Maas reach integrates months, not days.

## Network map (dashboard)

- One card at the top of the dashboard: a map with the reaches coloured by the fresh deposit's reported
  class and the modelled stations by the most likely class of their suspended sediment, next to a panel
  for the selected reach (click a reach or station): upstream regime, discharge entering the reach,
  suspended matter and its source, sediment load, metal flux, and the deposit's median, 90% range and
  class probabilities. Chosen over a schematic or reach cards because it reuses the station coordinates
  and shows where the reaches are.
- Reaches follow approximate waypoints along the rivers (`reaches[].path` in the config), placed by hand
  at towns on the river; display only, nothing is computed from them.
- Basemap: Sary's `open-street-map` raster tiles. Map labels use the basemap's own font, as map labels
  can only use fonts the tile server provides.

## Input assumptions for the reaches (researched, now used by the reach component)

Values for the reach and deposit components, from a literature search on 2026-11-19. [S] = read in the
source, [I] = inferred.

- Rhine split at the Pannerdensche Kop / IJsselkop, as fractions of Lobith discharge [S]: very low flow
  (680 m3/s) Waal 0.82, Nederrijn 0.02, IJssel 0.16; low (1300) 0.80 / 0.02 / 0.18; mean (2480) 0.69 /
  0.16 / 0.15; design flood (16,000) 0.635 / 0.211 / 0.154. Weir Driel keeps the IJssel at about
  285-350 m3/s at low flow. Suspended sediment divides roughly like the water (Frings 2005); bed load
  does not. Sources: Rijkswaterstaat waterverdeling maps (iplo.nl), Tienjarig Overzicht 1981-1990,
  verkenning afvoerverdeling Rijntakken (open.rijkswaterstaat.nl), Frings 2005 (dspace.library.uu.nl/handle/1874/11331).
- Distances [S]: Lobith km 862.2, IJsselkop km 878.5, Kampen km 994.5, so Lobith -> Kampen about 132 km.
  Eijsden km 2.6, Keizersveer km 247.6 on the Maas, so about 245 km. Sources: Rijkswaterstaat
  betrekkingslijnen Rijn 2022, kenmerkende waarden Maas 1991-2015.
- Travel times [I]: Lobith -> Kampen about 2 days at mean flow (1 at high, 3-4 at low). Eijsden ->
  Keizersveer about 1 week at mean flow, 2-3 days at high flow, 3-6 weeks at low flow (weir pools).
  KWR (BTO 2016.105) estimates travel time as length x cross-section / discharge.
- Tributaries [S]: between Eijsden and Keizersveer about 85 m3/s in total (Roer 23, Dommel 12.5, Niers 8,
  Aa 7, Geul 3.4) against about 230 m3/s in the Maas; the Geul carries Zn, Pb and Cd from old mines.
  Deltares (2008) found metal contents at Keizersveer generally lower than at Eijsden. IJssel tributaries
  are under about 5% of its mean flow.
- Fresh deposit [S/I]: Rijkswaterstaat / Deltares state that riverbed quality is set by the quality of
  the suspended matter that settles. No source gives a 30-day memory; deposition is episodic (floods)
  and harbours integrate months to years. Plan: keep 30 days as the default and test 14-90 days.
- Organic matter from organic carbon [S]: Dutch practice uses 1.724 (origin unclear); Van den Berg
  (H2O 1998, for STOWA) found 1.97 for freshwater sediments (literature range 1.8-2.4). The Regeling
  bodemkwaliteit asks for organic matter measured by loss on ignition (NEN 5754), not converted.
