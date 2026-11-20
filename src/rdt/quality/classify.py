"""Quality-class probabilities for suspended sediment.

Pipeline:
    MetalEstimate --sample--> metal draws --compare with soil-corrected limits--> class per draw
    --> share of draws per class = class probabilities

The limits in regulatory_limits.csv hold for a *standard soil* (25% lutum, 10% organic matter);
they must be corrected to the sample's own lutum and organic matter (Bijlage G):

    limit_corrected = limit_standard * (A + B * lutum + C * OM) / (A + B * 25 + C * 10)

with A, B, C from soil_correction.csv and lutum / OM clipped to the minimum/maximum listed there.
Correcting the limits is equivalent to correcting the concentration to standard soil (the Bijlage G
formulation): both scale one side of the comparison by the same factor.
NOTE: the lab reports organic *carbon* (corg); organic matter needs a conversion
(commonly OM ~ 1.7-2 x corg) - choose and document a factor.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from rdt.softsensor.estimate import MetalEstimate

QUALITY_CLASSES: tuple[str, ...] = ("clean", "slightly contaminated", "moderately contaminated", "heavily contaminated")


def _clip(value: float, lo: float, hi: float) -> float:
    """Clip to [lo, hi]; a NaN bound (empty in soil_correction.csv) means no bound."""
    if pd.notna(lo):
        value = max(value, float(lo))
    if pd.notna(hi):
        value = min(value, float(hi))
    return value


def corrected_limits(
    limits: pd.DataFrame,
    soil_correction: pd.DataFrame,
    parameter: str,
    lutum_pct: float,
    om_pct: float,
) -> pd.DataFrame:
    """Class boundaries for one substance (e.g. 'Cd'), corrected to the given lutum / OM.

    Returns rows (class_en, lower_exclusive, upper_inclusive) in QUALITY_CLASSES order.
    """
    if not (np.isfinite(lutum_pct) and np.isfinite(om_pct)):
        raise ValueError(f"Soil-type correction needs lutum and organic matter, got lutum={lutum_pct}, "
                         f"organic matter={om_pct}.")
    rows = limits[limits["parameter"] == parameter].set_index("class_en")
    if set(rows.index) != set(QUALITY_CLASSES):
        raise ValueError(f"regulatory limits for '{parameter}' do not list the classes {QUALITY_CLASSES}.")
    c = soil_correction.loc[parameter]
    lutum = _clip(lutum_pct, c["lutum_min_pct"], c["lutum_max_pct"])
    om = _clip(om_pct, c["om_min_pct"], c["om_max_pct"])
    factor = (c["A"] + c["B"] * lutum + c["C"] * om) / (
        c["A"] + c["B"] * c["standard_lutum_pct"] + c["C"] * c["standard_om_pct"])
    out = rows.loc[list(QUALITY_CLASSES), ["lower_exclusive", "upper_inclusive"]].astype(float) * factor
    return out.rename_axis("class_en").reset_index()


def assign_class(concentration: float | np.ndarray, limits: pd.DataFrame) -> np.ndarray:
    """Index into QUALITY_CLASSES for each concentration, given rows from `corrected_limits`.

    A class holds when lower < concentration <= upper, so a value equal to a limit belongs to the
    cleaner class; searchsorted with side="left" gives exactly that.
    """
    uppers = limits["upper_inclusive"].to_numpy(float)[: len(QUALITY_CLASSES) - 1]
    return np.searchsorted(uppers, np.asarray(concentration, float), side="left")


def sample_class(
    concentrations: dict[str, float],
    limits: pd.DataFrame,
    soil_correction: pd.DataFrame,
    lutum_pct: float,
    om_pct: float,
) -> int:
    """Index into QUALITY_CLASSES of a measured sample: the worst class over its metals ({'cd': mg/kg, ...})."""
    return max(
        int(assign_class(value, corrected_limits(limits, soil_correction, metal.capitalize(), lutum_pct, om_pct)))
        for metal, value in concentrations.items()
    )


def _class_draws(
    estimate: MetalEstimate,
    limits: pd.DataFrame,
    soil_correction: pd.DataFrame,
    lutum_pct: float,
    om_pct: float,
    n_draws: int,
    rng: np.random.Generator | None,
) -> np.ndarray:
    """Class index per joint draw and metal, shape (n_draws, n_metals)."""
    if not np.all(np.isfinite(estimate.mean_log)):
        missing = [m for m, v in zip(estimate.metals, estimate.mean_log) if not np.isfinite(v)]
        raise ValueError(f"Estimate at {estimate.station} has no value for {missing}; cannot classify.")
    return _classes_of_draws(estimate.sample(n_draws, rng), estimate.metals, limits, soil_correction,
                             lutum_pct, om_pct)


def _classes_of_draws(
    draws: np.ndarray,
    metals: Sequence[str],
    limits: pd.DataFrame,
    soil_correction: pd.DataFrame,
    lutum_pct: float,
    om_pct: float,
) -> np.ndarray:
    """Class index per draw and metal for draws in mg/kg dw, shape (n_draws, n_metals)."""
    classes = np.empty(draws.shape, dtype=int)
    for j, metal in enumerate(metals):
        cl = corrected_limits(limits, soil_correction, metal.capitalize(), lutum_pct, om_pct)
        classes[:, j] = assign_class(draws[:, j], cl)
    return classes


def _shares(classes: np.ndarray) -> dict[str, float]:
    counts = np.bincount(classes, minlength=len(QUALITY_CLASSES))
    return dict(zip(QUALITY_CLASSES, (counts / len(classes)).tolist(), strict=True))


def class_probabilities(
    estimate: MetalEstimate,
    limits: pd.DataFrame,
    soil_correction: pd.DataFrame,
    lutum_pct: float,
    om_pct: float,
    n_draws: int = 5000,
    rng: np.random.Generator | None = None,
) -> dict[str, dict[str, float]]:
    """{metal: {class: probability}} by Monte-Carlo over `estimate.sample(n_draws)`.

    Lutum and organic matter are treated as known (in practice: from the last lab sample).
    """
    classes = _class_draws(estimate, limits, soil_correction, lutum_pct, om_pct, n_draws, rng)
    return {m: _shares(classes[:, j]) for j, m in enumerate(estimate.metals)}


def sample_class_probabilities(
    estimate: MetalEstimate,
    limits: pd.DataFrame,
    soil_correction: pd.DataFrame,
    lutum_pct: float,
    om_pct: float,
    n_draws: int = 5000,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """{class: probability} for the sediment as a whole: per draw, the worst class over all metals.

    Uses the joint draws (not per-metal probabilities) so correlation between metals is respected.
    """
    classes = _class_draws(estimate, limits, soil_correction, lutum_pct, om_pct, n_draws, rng)
    return _shares(classes.max(axis=1))


def draws_class_probabilities(
    draws: np.ndarray,
    metals: Sequence[str],
    limits: pd.DataFrame,
    soil_correction: pd.DataFrame,
    lutum_pct: float,
    om_pct: float,
) -> dict[str, float]:
    """{class: probability} for joint draws in mg/kg dw (n_draws, n_metals): per draw, the worst class."""
    return _shares(_classes_of_draws(draws, metals, limits, soil_correction, lutum_pct, om_pct).max(axis=1))


def reuse_applications(quality_class: str) -> list[str]:
    """Reuse applications not ruled out by metal content for a given class.

    TODO: fill in from the Regeling bodemkwaliteit / Besluit bodemkwaliteit (e.g. which soil-function
    classes and large-scale applications each class may be used for). Keep the source reference here.
    """
    raise NotImplementedError
