# River Digital Twin

Digital twin of heavy-metal transport via suspended sediment in the Rhine (Lobith) and Maas (Eijsden).

**Group 13:** Nicholas Miot, Jacob Shashoua, Sary Abou

## Purpose

Heavy metal content in river sediment is only measured by periodic lab analysis (weekly at Eijsden, fortnightly at Lobith). This project builds a digital twin that continuously estimates the metal content of suspended sediment between lab samples, translates it into the most likely quality class under the Dutch *Regeling bodemkwaliteit 2022*, and indicates which reuse applications are not ruled out by metal content.

## Planned functionality

- Replay of historical monitoring data as live per-sensor streams through a message broker
- Representation model of the river network: stations, reaches, hydrological state, sediment and metal fluxes
- Machine-learning soft sensor (trained only on data before the replayed period) estimating metal content with uncertainty, converted to quality-class probabilities
- Exploratory: reinforcement learning for adaptive lab sampling
- Live comparison of predictions against incoming lab results, with drift detection
- Dashboard: map of reaches coloured by predicted class, time series with uncertainty bands, alerts

## Data

Rijkswaterstaat WaterWebservices (public), 2012–2025: 40 sensors at four stations — turbidity, discharge, suspended sediment concentration, and lab analyses of metals, grain size and organic carbon. About 1.68 million readings.

- https://rijkswaterstaatdata.nl/waterdata/
- https://waterinfo.rws.nl/nav/bulkdownload

The dataset (~185 MB) is **not in git**: get the `dataset` folder from the group's shared drive and put it at `data/raw/dataset/`, so that `data/raw/dataset/Full/readings.csv` exists. `data/raw/dataset/DATASET.md` (tracked) describes the files. The code only reads `Full/`; `Divided/` is the same data split by year. The project proposal is `data/raw/Group13_Project_Dataset.docx`.

## Getting started

Requires Python 3.11+. Put the dataset in place (see Data above), then:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

rdt info                              # summarise the dataset
rdt run                               # replay 2021–2024 at 1 simulated day/s + dashboard on http://127.0.0.1:8050
rdt run --speed 604800                # 1 simulated week per second
rdt run --no-dashboard --speed 0      # headless, as fast as possible
pytest                                # tests
```

Settings (time split, replay speed, soft-sensor type, stations) are in `config/default.yaml`.

## Project layout

```
config/          YAML configuration
src/rdt/         the twin (see docs/ARCHITECTURE.md)
tests/           pytest suite (synthetic data; real-data checks skip if the dataset is absent)
analysis/        statistical analysis: 01 correlations, 02 trend test, 03 model comparison,
                 04 soft-sensor evaluation (2021–2024), 05 input tuning on a 2018–2020 validation split
                 (outputs in analysis/results/)
docs/            architecture notes
data/raw/        project proposal; dataset goes in data/raw/dataset/ (not in git)
```
