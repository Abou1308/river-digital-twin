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

Raw data goes in `data/raw/`. The project proposal is `data/raw/Group13_Project_Dataset.docx`.
