# Dataset

Cleaned Rijkswaterstaat data for Eijsden (Maas) and Lobith (Rhine), 2012 to 2025. `readings.csv` is the main dataset: one row per sensor reading (time, sensor, value). `sensors.csv` explains what each sensor is and its unit. `model_table.csv` has one row per lab sample, with the metal concentrations and turbidity/discharge features ready for modelling. `regulatory_limits.csv` and `soil_correction.csv` are the Dutch soil-quality limits and the correction constants that go with them. `Full/` has every file whole. `Divided/` has the same data split by year, with the three tables that don't change by year in `Divided/reference/`.

So far the timestamps have been converted to UTC, and gap values and duplicates have been removed. Nothing has been averaged or resampled. Values below a detection limit are kept at the limit with `censored = True`, and questionable readings are kept but flagged in the `qc` column.

