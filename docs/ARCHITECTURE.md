# Architecture

```
 readings.csv ──► Replayer ──► Broker ──► DigitalTwin ──► StateStore ──► Dashboard
 (2021–2024)     (SimClock)   (topics      │                (readings,     (Dash, polls
                               rws/...)    │                 estimates,     the store)
                                           │                 lab vs pred,
                                           │                 alerts)
                                           ├─ SoftSensor        predict / assimilate lab results
                                           ├─ FeatureBuilder    live window features
                                           ├─ DriftDetector     lab result vs prediction
                                           ├─ quality           class probabilities, reuse options
                                           └─ SamplingPolicy    (exploratory) when to sample
```

## Packages (`src/rdt/`)

| Package | Role | Status |
|---|---|---|
| `config.py` | Typed YAML config (`config/default.yaml`), validates the train/replay split | done |
| `data/` | CSV loaders; `SensorInfo` catalogue derived from `sensors.csv` | done |
| `messaging/` | `SensorMessage` (JSON), `Broker` protocol, in-memory broker with MQTT wildcards | in-memory done, MQTT stub |
| `replay/` | `SimClock` (speed factor or as-fast-as-possible) and `Replayer` | done |
| `twin/network.py` | Representation model: stations, reaches | structure done; reach state in `representation/` |
| `twin/components.py` | Hook for optional representation components (`twin.components`), called on each reading and estimate | done |
| `representation/` | Optional components: sensor health, turbidity-SPM check, hydrological regime, reach flux and fresh deposit, downstream check (decisions in `docs/DECISIONS.md`) | done; no dashboard view yet |
| `twin/twin.py` | Orchestrator: routes messages, compares lab vs prediction, assimilates, estimates | done |
| `state.py` | `StateStore` protocol + thread-safe in-memory store; the twin/dashboard boundary | done |
| `features/` | Sensor-window inputs (only those in `softsensor.features`), same code offline and live; verified against `model_table.csv` | done |
| `softsensor/` | `SoftSensor` contract, `MetalEstimate` (joint log-normal); Kalman (main), ridge_log and persistence (references); online range recalibration wrapper (on by default); `build_table` | done; evaluated in `analysis/04_soft_sensor_evaluation.py`, extra inputs tested in `analysis/05_model_tuning.py` (no gain) |
| `quality/` | Soil-type correction, class probabilities, reuse applications | correction and class probabilities done (tests in `tests/test_quality.py`); reuse applications TODO |
| `monitoring/` | Drift detection (CUSUM default, interval exceedance), evaluation metrics, class evaluation (`ClassComparison`), calibration checks, online Platt recalibration (analysis only) | done; calibration in `analysis/06`-`07` |
| `sampling/` | Sampling policies (RL track) | fixed-interval done; RL TODO |
| `dashboard/` | Dash app + pure plotly figure builders | metal series with reference line, scoreboard, sensors, quality-class panel (current class probabilities and class forecast scores per station), alerts |
| `runtime.py` | `build_system(cfg)` wires everything | done |
| `cli.py` | `rdt info`, `rdt run` | done |

## Design rules

1. **One leakage boundary.** Only `runtime.build_system` decides what is training data (`< periods.train_end`) and what is replayed (`>= periods.replay_start`). `SoftSensor.fit` re-checks with `check_no_leakage`.
2. **Everything in log space, jointly.** Soft sensors return a `MetalEstimate` (mean and covariance of ln concentration for all metals). Intervals, class probabilities and drift scores all derive from it, so any model that fills this contract plugs into the whole twin.
3. **Predict before assimilate.** When a lab result arrives, the twin first records the prediction made without it (`LabComparison`, drift detector), then updates the model. That's what makes the live evaluation fair.
4. **Dashboard reads only the `StateStore`.** It never touches the twin, so the twin can later run in a separate process (MQTT + a persistent store) without changing the dashboard.
5. **Topics come from `sensors.csv`.** The broker can be swapped for MQTT without changing publishers or subscribers.

## Where to start building

- **Soft sensor:** done (Kalman, see `softsensor/kalman.py` docstring). Lab results are streamed after an assumed turnaround (`softsensor.lab_delay_d`); the twin compares each result with its prediction for the sample time made with only the results known then.
- **Quality:** `quality/classify.py`: reuse mapping (`reuse_applications`).
- **Representation model:** components in `representation/`; the reach and downstream components cover fluxes, travel time and the Keizersveer / Kampen check.
- **Dashboard:** reaches coloured by class, once the representation model propagates fluxes.
