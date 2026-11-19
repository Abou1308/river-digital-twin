"""CUSUM drift detector, ported from the MVP (MVP/twin/model.py, `Cusum`).

Two-sided tabular CUSUM (Page) on the standardised innovations z of incoming lab results, kept per
station and metal:

    up   = max(0, up   + z - delta)        down = max(0, down - z - delta)

An alarm is raised when either sum exceeds `threshold`; both sums then reset. Unlike
`IntervalExceedanceDetector`, which needs several large misses in a row, CUSUM also detects a small but
persistent bias. The defaults are the textbook settings (delta = 0.5 sd, threshold = 5 sd), as in the MVP.
"""

from __future__ import annotations

from rdt.monitoring.drift import DriftDetector, standardised_innovation
from rdt.softsensor.estimate import MetalEstimate
from rdt.state import Alert


class CusumDetector(DriftDetector):
    def __init__(self, delta: float = 0.5, threshold: float = 5.0) -> None:
        self.delta = delta
        self.threshold = threshold
        self._sums: dict[tuple[str, str], tuple[float, float]] = {}

    def observe(self, prior: MetalEstimate, metal: str, observed: float, censored: bool) -> Alert | None:
        if censored or observed <= 0:
            return None
        z = standardised_innovation(prior, metal, observed)
        key = (prior.station, metal)
        up, down = self._sums.get(key, (0.0, 0.0))
        up, down = max(0.0, up + z - self.delta), max(0.0, down - z - self.delta)
        direction = "up" if up > self.threshold else "down" if down > self.threshold else None
        if direction is None:
            self._sums[key] = (up, down)
            return None
        self._sums[key] = (0.0, 0.0)
        side = "above" if direction == "up" else "below"
        return Alert(
            time_utc=prior.time_utc,
            station=prior.station,
            metal=metal,
            kind="drift",
            severity="warning",
            message=f"{metal.upper()} lab results persistently {side} prediction (CUSUM {direction}, last z={z:+.1f})",
        )
