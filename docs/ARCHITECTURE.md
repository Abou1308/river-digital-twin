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
| `messaging/` | `SensorMessage` (JSON), `Broker` protocol, in-memory broker with MQTT wildcards, MQTT broker (paho-mqtt) with status messages between processes | done |
| `replay/` | `SimClock` (speed factor or as-fast-as-possible) and `Replayer` | done |
| `twin/network.py` | Representation model: stations, reaches | structure done; reach state in `representation/` |
| `twin/components.py` | Hook for optional representation components (`twin.components`), called on each reading and estimate | done |
| `representation/` | Optional components: sensor health, turbidity-SPM check, hydrological regime, reach flux and fresh deposit, downstream check (decisions in `docs/DECISIONS.md`) | done; reaches shown on the network map |
| `twin/twin.py` | Orchestrator: routes messages, compares lab vs prediction, assimilates, estimates | done |
| `state.py` | `StateStore` protocol + thread-safe in-memory store; the twin/dashboard boundary | done |
| `features/` | Sensor-window inputs (only those in `softsensor.features`), same code offline and live; verified against `model_table.csv` | done |
| `softsensor/` | `SoftSensor` contract, `MetalEstimate` (joint log-normal); Kalman (main), ridge_log and persistence (references); online range recalibration wrapper (on by default); `build_table` | done; evaluated in `analysis/04_soft_sensor_evaluation.py`, extra inputs tested in `analysis/05_model_tuning.py` (no gain) |
| `quality/` | Soil-type correction, class probabilities, reuse applications | correction and class probabilities done (tests in `tests/test_quality.py`); reuse applications TODO |
| `monitoring/` | Drift detection (CUSUM default, interval exceedance), evaluation metrics, class evaluation (`ClassComparison`), calibration checks, online Platt recalibration (analysis only) | done; calibration in `analysis/06`-`07` |
| `sampling/` | Sampling policies (RL track) | fixed-interval done; RL TODO |
| `dashboard/` | Dash app + pure plotly figure builders | network map with reach details, metal series with reference line, scoreboard, sensors, quality-class panel (current class probabilities and class forecast scores per station), alerts |
| `runtime.py` | `build_twin` (fit and build the twin), `build_replayer` (the data source), `build_system` (both in one process) | done |
| `cli.py` | `rdt info`, `rdt run` (one process), `rdt twin` and `rdt replay` (two processes over MQTT) | done |

## Design rules

1. **One leakage boundary.** Only `runtime` (`build_twin` and `build_replayer`) decides what is training data (`< periods.train_end`) and what is replayed (`>= periods.replay_start`). `SoftSensor.fit` re-checks with `check_no_leakage`.
2. **Everything in log space, jointly.** Soft sensors return a `MetalEstimate` (mean and covariance of ln concentration for all metals). Intervals, class probabilities and drift scores all derive from it, so any model that fills this contract plugs into the whole twin.
3. **Predict before assimilate.** When a lab result arrives, the twin first records the prediction made without it (`LabComparison`, drift detector), then updates the model. That's what makes the live evaluation fair.
4. **Dashboard reads only the `StateStore`.** It never touches the twin, so the dashboard could later run in its own process (with a persistent store) without changing it. The replay already runs separately (`rdt replay`, see Deployment).
5. **Topics come from `sensors.csv`.** The broker can be swapped for MQTT without changing publishers or subscribers.

## Where to start building

- **Soft sensor:** done (Kalman, see `softsensor/kalman.py` docstring). Lab results are streamed after an assumed turnaround (`softsensor.lab_delay_d`); the twin compares each result with its prediction for the sample time made with only the results known then.
- **Quality:** `quality/classify.py`: reuse mapping (`reuse_applications`).
- **Representation model:** components in `representation/`; the reach and downstream components cover fluxes, travel time and the Keizersveer / Kampen check.
- **Dashboard:** network map done; the downstream check's comparisons are not shown yet.

## Deployment

`rdt run` keeps everything in one process with the in-memory broker. With MQTT, the data source and the
twin are separate programs, as sensors and a twin would be:

```
 rdt replay ──MQTT──► Mosquitto ──MQTT──► rdt twin (twin + StateStore + dashboard :8050)
            rws/<station>/...
            rdt/status/replay = finished        rdt/status/twin = starting | ready | offline (retained)
```

The twin announces `ready` once it is subscribed; the replay waits for that before publishing, so no
reading is lost. `compose.yaml` runs the three as containers (`config/docker.yaml`); `config/mqtt.yaml`
runs the two programs on the host against a broker on localhost.
