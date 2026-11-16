"""Quality-class probabilities for suspended sediment.

Pipeline:
    MetalEstimate --sample--> metal draws --compare with soil-corrected limits--> class per draw
    --> share of draws per class = class probabilities

The limits in regulatory_limits.csv hold for a *standard soil* (25% lutum, 10% organic matter);
they must be corrected to the sample's own lutum and organic matter (Bijlage G):

    limit_corrected = limit_standard * (A + B * lutum + C * OM) / (A + B * 25 + C * 10)

with A, B, C from soil_correction.csv and lutum / OM clipped to the minimum/maximum listed there.
NOTE: the lab reports organic *carbon* (corg); organic matter needs a conversion
(commonly OM ~ 1.7-2 x corg) - choose and document a factor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rdt.softsensor.estimate import MetalEstimate

QUALITY_CLASSES: tuple[str, ...] = ("clean", "slightly contaminated", "moderately contaminated", "heavily contaminated")


def corrected_limits(
    limits: pd.DataFrame,
    soil_correction: pd.DataFrame,
    parameter: str,
    lutum_pct: float,
    om_pct: float,
) -> pd.DataFrame:
    """Class boundaries for one substance (e.g. 'Cd'), corrected to the given lutum / OM.

    Returns rows (class_en, lower_exclusive, upper_inclusive) in QUALITY_CLASSES order.
    TODO: implement the formula in the module docstring.
    """
    raise NotImplementedError


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

    TODO: implement; later also propagate uncertainty in lutum / OM (only known from the last lab sample).
    """
    raise NotImplementedError


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
    TODO: implement.
    """
    raise NotImplementedError


def reuse_applications(quality_class: str) -> list[str]:
    """Reuse applications not ruled out by metal content for a given class.

    TODO: fill in from the Regeling bodemkwaliteit / Besluit bodemkwaliteit (e.g. which soil-function
    classes and large-scale applications each class may be used for). Keep the source reference here.
    """
    raise NotImplementedError
