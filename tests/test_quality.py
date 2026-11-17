"""Soil-type correction (Rbk 2022 Bijlage G), class assignment (Bijlage B tabel 2) and class probabilities."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from rdt.data.schema import METALS
from rdt.quality import (
    QUALITY_CLASSES,
    assign_class,
    class_probabilities,
    corrected_limits,
    sample_class,
    sample_class_probabilities,
)
from rdt.softsensor import MetalEstimate

nan = float("nan")

# Rows copied from regulatory_limits.csv and soil_correction.csv (Rbk 2022, toestand 2026-01-31).
LIMITS = pd.DataFrame(
    [("Cd", "clean", nan, 0.6), ("Cd", "slightly contaminated", 0.6, 4),
     ("Cd", "moderately contaminated", 4, 14), ("Cd", "heavily contaminated", 14, nan),
     ("Zn", "clean", nan, 140), ("Zn", "slightly contaminated", 140, 563),
     ("Zn", "moderately contaminated", 563, 2000), ("Zn", "heavily contaminated", 2000, nan)],
    columns=["parameter", "class_en", "lower_exclusive", "upper_inclusive"],
)
SOIL = pd.DataFrame(
    [("Cd", 0.4, 0.007, 0.021, 2, nan, 2, nan, 25, 10), ("Zn", 50, 3, 1.5, 2, nan, 2, nan, 25, 10)],
    columns=["parameter", "A", "B", "C", "lutum_min_pct", "lutum_max_pct", "om_min_pct", "om_max_pct",
             "standard_lutum_pct", "standard_om_pct"],
).set_index("parameter")


def _uppers(parameter: str, lutum: float, om: float) -> list[float]:
    return corrected_limits(LIMITS, SOIL, parameter, lutum, om)["upper_inclusive"].tolist()[:3]


def _estimate(medians: dict[str, float], sd: float, corr: float = 0.0) -> MetalEstimate:
    n = len(medians)
    cov = sd**2 * (np.full((n, n), corr) + (1 - corr) * np.eye(n))
    return MetalEstimate("eijsden", datetime(2022, 1, 1, tzinfo=timezone.utc), tuple(medians),
                         np.log(list(medians.values())), cov)


def test_standard_soil_leaves_limits_unchanged():
    cl = corrected_limits(LIMITS, SOIL, "Zn", 25, 10)
    assert cl["class_en"].tolist() == list(QUALITY_CLASSES)
    assert cl["upper_inclusive"].tolist()[:3] == pytest.approx([140, 563, 2000])
    assert cl["lower_exclusive"].tolist()[1:] == pytest.approx([140, 563, 2000])


def test_zinc_worked_example():
    # Zn: A = 50, B = 3, C = 1.5. Lutum 30%, organic matter 12%.
    # factor = (50 + 3*30 + 1.5*12) / (50 + 3*25 + 1.5*10) = 158 / 140
    assert _uppers("Zn", 30, 12) == pytest.approx([158.0, 635.385714, 2257.142857])


def test_cadmium_below_minimum_is_clipped():
    # Cd: A = 0.4, B = 0.007, C = 0.021. Lutum 1% and organic matter 1.5% are below the 2% minimum,
    # so L = H = 2: factor = (0.4 + 0.014 + 0.042) / (0.4 + 0.175 + 0.21) = 0.456 / 0.785
    assert _uppers("Cd", 1, 1.5) == pytest.approx([0.348535, 2.323567, 8.132484], abs=1e-6)
    assert _uppers("Cd", 1, 1.5) == _uppers("Cd", 2, 2)


def test_correcting_limits_equals_correcting_concentration():
    # 630 mg/kg Zn in a soil with 30% lutum and 12% organic matter: standard-soil concentration
    # 630 * 140/158 = 558.2 is slightly contaminated; so is 630 against the corrected limits.
    corrected = corrected_limits(LIMITS, SOIL, "Zn", 30, 12)
    standard = corrected_limits(LIMITS, SOIL, "Zn", 25, 10)
    assert assign_class(630, corrected) == assign_class(630 * 140 / 158, standard) == 1
    assert assign_class(630, standard) == 2          # without the correction the class would be worse


def test_class_boundaries_are_lower_exclusive_upper_inclusive():
    zn = corrected_limits(LIMITS, SOIL, "Zn", 25, 10)
    cases = {139.9: 0, 140.0: 0, 140.001: 1, 563.0: 1, 563.001: 2, 2000.0: 2, 2000.001: 3}
    for conc, expected in cases.items():
        assert assign_class(conc, zn) == expected, conc
    cd = corrected_limits(LIMITS, SOIL, "Cd", 25, 10)
    assert assign_class([0.6, 0.6001, 4, 4.0001, 14, 14.0001], cd).tolist() == [0, 1, 1, 2, 2, 3]


def test_missing_soil_properties_raise():
    with pytest.raises(ValueError, match="lutum"):
        corrected_limits(LIMITS, SOIL, "Zn", nan, 10)
    with pytest.raises(ValueError, match="organic matter"):
        corrected_limits(LIMITS, SOIL, "Zn", 25, nan)


def test_estimate_without_value_raises():
    est = _estimate({"cd": 1.0, "zn": 300.0}, sd=0.1)
    est = MetalEstimate(est.station, est.time_utc, est.metals, np.array([np.nan, np.log(300)]), est.cov_log)
    with pytest.raises(ValueError, match="cd"):
        class_probabilities(est, LIMITS, SOIL, 25, 10)


def test_class_probabilities_per_metal():
    rng = np.random.default_rng(0)
    p = class_probabilities(_estimate({"cd": 6.0, "zn": 300.0}, sd=0.01), LIMITS, SOIL, 25, 10, rng=rng)
    assert p["cd"]["moderately contaminated"] == 1.0
    assert p["zn"]["slightly contaminated"] == 1.0
    # median exactly on the clean / slightly boundary: half the draws on each side
    p = class_probabilities(_estimate({"zn": 140.0}, sd=0.1), LIMITS, SOIL, 25, 10, n_draws=20000, rng=rng)
    assert p["zn"]["clean"] == pytest.approx(0.5, abs=0.02)
    assert sum(p["zn"].values()) == pytest.approx(1.0)


def test_overall_class_is_worst_metal_from_joint_draws():
    """Cd and Zn both on their clean / slightly boundary. If they move together, the sediment is
    slightly contaminated about half the time; if independent, three quarters of the time."""
    medians = {"cd": 0.6, "zn": 140.0}
    rng = np.random.default_rng(1)
    together = sample_class_probabilities(_estimate(medians, 0.1, corr=0.999), LIMITS, SOIL, 25, 10,
                                          n_draws=20000, rng=rng)
    independent = sample_class_probabilities(_estimate(medians, 0.1), LIMITS, SOIL, 25, 10,
                                             n_draws=20000, rng=rng)
    assert together["slightly contaminated"] == pytest.approx(0.5, abs=0.02)
    assert independent["slightly contaminated"] == pytest.approx(0.75, abs=0.02)


def test_real_limits_cover_modelled_metals(cfg):
    if not (cfg.data_dir / "regulatory_limits.csv").exists():
        pytest.skip("dataset not available")
    from rdt.data import load_regulatory_limits, load_soil_correction

    limits, soil = load_regulatory_limits(cfg.data_dir), load_soil_correction(cfg.data_dir)
    for metal in METALS:
        uppers = corrected_limits(limits, soil, metal.capitalize(), 25, 10)["upper_inclusive"].to_numpy()[:3]
        assert np.all(np.isfinite(uppers)) and np.all(np.diff(uppers) > 0), metal


def test_sample_class_is_worst_metal():
    # Cd 0.5 is clean, Zn 600 is moderately contaminated (standard soil).
    assert sample_class({"cd": 0.5, "zn": 600.0}, LIMITS, SOIL, 25, 10) == 2
    assert sample_class({"cd": 0.5}, LIMITS, SOIL, 25, 10) == 0
