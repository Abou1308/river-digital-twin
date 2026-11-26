"""Reuse of the sediment as dredged material (baggerspecie), as far as the quality class decides it.

The classes of `QUALITY_CLASSES` are the classes for baggerspecie and waterbodem of the Regeling
bodemkwaliteit 2022, bijlage B, tabel 2 (niet verontreinigd / algemeen toepasbaar, licht, matig and sterk
verontreinigd). What they allow is set by the Besluit activiteiten leefomgeving (Bal), version of
1 July 2026, read from the official text (repository.officiele-overheidspublicaties.nl, BWBR0041330):

- Bal 4.1272, second paragraph: baggerspecie applied in a surface water (oppervlaktewaterlichaam) must
  (a) meet the quality requirements of the class of the receiving water bottom, and (c) those of the
  class matig verontreinigd. Water bottoms are classed niet, licht or matig verontreinigd (Besluit
  bodemkwaliteit 25d, fourth paragraph). So clean material may go onto any water bottom, licht
  verontreinigd onto a licht or matig verontreinigd one, matig verontreinigd onto a matig one only.
- Bal 4.1273: sterk verontreinigd material may only be applied by maatwerk, within a designated
  bodembeheergebied, taken from and applied onto locations that are already diffusely sterk verontreinigd.
- Applying it on land (Bal 4.1272, first paragraph) is judged against the land classes (landbouw/natuur,
  wonen, industrie; Rbk 2022 bijlage B tabel 1) and the land use of the receiving site; spreading
  (verspreiden, Bal 4.1269 third paragraph) against separate tables (tabel 3b, 3c). Neither follows from
  this class, so neither is assessed here.

The twin estimates five metals (Cd, Cu, Ni, Pb, Zn). The law classifies on all listed substances, so a
class from the twin rules uses out; it does not rule them in. Concentrations are compared after
correction to standard soil (Rbk 2022, as in classify.py).
"""

from __future__ import annotations

from collections.abc import Mapping

from rdt.quality.classify import QUALITY_CLASSES

# Classes a water bottom can have (Besluit bodemkwaliteit 25d(4)(b)); heavily contaminated is not one.
RECEIVING_WATER_BOTTOMS: tuple[str, ...] = QUALITY_CLASSES[:3]
_NOT_ON_LAND = ("Applying it on land or spreading it is judged against other tables (Rbk 2022 tabel 1, "
                "3b, 3c) and is not assessed here.")


def _check(quality_class: str) -> int:
    if quality_class not in QUALITY_CLASSES:
        raise ValueError(f"Unknown quality class {quality_class!r}; expected one of {QUALITY_CLASSES}.")
    return QUALITY_CLASSES.index(quality_class)


def reuse_applications(quality_class: str) -> list[str]:
    """Uses of dredged material of this class that the class does not rule out (module docstring)."""
    i = _check(quality_class)
    if i == len(QUALITY_CLASSES) - 1:
        uses = ["Not applicable in surface water; only by maatwerk from and onto locations that are already "
                "diffusely heavily contaminated (Bal 4.1273)."]
    else:
        bottoms = " or ".join(RECEIVING_WATER_BOTTOMS[i:])
        where = "any water bottom" if i == 0 else f"a water bottom classed {bottoms}"
        uses = [f"Applicable in surface water on {where} (Bal 4.1272)."]
    return uses + [_NOT_ON_LAND]


def surface_water_probabilities(probabilities: Mapping[str, float]) -> dict[str, float]:
    """Probability that the material may be applied in surface water, per class of the receiving water
    bottom, from class probabilities: material may go onto a bottom of its own class or worse."""
    for cls in probabilities:
        _check(cls)
    return {bottom: sum(probabilities.get(c, 0.0) for c in QUALITY_CLASSES[:i + 1])
            for i, bottom in enumerate(RECEIVING_WATER_BOTTOMS)}
