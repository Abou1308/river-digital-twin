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


class Cusum:
    """One two-sided CUSUM; `update` returns "up", "down" or None and resets both sums after an alarm."""

    def __init__(self, delta: float = 0.5, threshold: float = 5.0) -> None:
        self.delta = delta
        self.threshold = threshold
        self.up = self.down = 0.0

    def update(self, z: float) -> str | None:
        self.up = max(0.0, self.up + z - self.delta)
        self.down = max(0.0, self.down - z - self.delta)
        direction = "up" if self.up > self.threshold else "down" if self.down > self.threshold else None
        if direction is not None:
            self.up = self.down = 0.0
        return direction


class CusumDetector(DriftDetector):
    def __init__(self, delta: float = 0.5, threshold: float = 5.0) -> None:
        self.delta = delta
        self.threshold = threshold
        self._sums: dict[tuple[str, str], Cusum] = {}

    def observe(self, prior: MetalEstimate, metal: str, observed: float, censored: bool) -> Alert | None:
        if censored or observed <= 0:
            return None
        z = standardised_innovation(prior, metal, observed)
        cusum = self._sums.setdefault((prior.station, metal), Cusum(self.delta, self.threshold))
        direction = cusum.update(z)
        if direction is None:
            return None
        side = "above" if direction == "up" else "below"
        return Alert(
            time_utc=prior.time_utc,
            station=prior.station,
            metal=metal,
            kind="drift",
            severity="warning",
            message=f"{metal.upper()} lab results persistently {side} prediction (CUSUM {direction}, last z={z:+.1f})",
        )
