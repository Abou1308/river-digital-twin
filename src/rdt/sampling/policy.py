"""Policies that decide when to take a lab sample (exploratory reinforcement-learning track).

Evaluation idea: replay history but only *reveal* a lab result to the soft sensor when the policy
asks for a sample (and one exists near that time); score = estimation/classification error vs
number of samples used. The status quo is `FixedIntervalPolicy` (weekly Eijsden, fortnightly Lobith).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta

from rdt.softsensor.estimate import MetalEstimate


class SamplingPolicy(ABC):
    @abstractmethod
    def should_sample(self, station: str, t: datetime, estimate: MetalEstimate | None) -> bool:
        """Return True if a lab sample should be taken at `station` now."""


class FixedIntervalPolicy(SamplingPolicy):
    def __init__(self, interval: timedelta) -> None:
        self.interval = interval
        self._last: dict[str, datetime] = {}

    def should_sample(self, station: str, t: datetime, estimate: MetalEstimate | None) -> bool:
        last = self._last.get(station)
        if last is None or t - last >= self.interval:
            self._last[station] = t
            return True
        return False


# TODO: UncertaintyThresholdPolicy (sample when class probability is ambiguous / sd_log large) as a
# strong heuristic baseline, then an RL agent (state: estimate + hydrology, action: sample or not,
# reward: -classification error - cost per sample).
