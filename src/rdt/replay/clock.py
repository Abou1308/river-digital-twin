"""Simulated clock mapping wall-clock time onto historical (simulated) time."""

from __future__ import annotations

import threading
import time

import pandas as pd


class SimClock:
    """Simulated time = `start` + (wall seconds elapsed) * `speed`.

    With ``speed=None`` the clock never waits: the replay runs as fast as possible and `now()`
    simply reports the time of the most recent event (useful for tests and batch evaluation).
    """

    def __init__(self, start: pd.Timestamp, speed: float | None = None) -> None:
        self.start = start
        self.speed = speed
        self._wall_start = time.monotonic()
        self._last_event = start

    def now(self) -> pd.Timestamp:
        if self.speed is None:
            return self._last_event
        return self.start + pd.Timedelta(seconds=(time.monotonic() - self._wall_start) * self.speed)

    def wait_until(self, sim_time: pd.Timestamp, stop: threading.Event | None = None) -> bool:
        """Block until simulated time reaches `sim_time`. Returns False if `stop` was set."""
        self._last_event = max(self._last_event, sim_time)
        if self.speed is None:
            return not (stop and stop.is_set())
        delay = (sim_time - self.now()).total_seconds() / self.speed
        if delay <= 0:
            return not (stop and stop.is_set())
        if stop is not None:
            return not stop.wait(delay)
        time.sleep(delay)
        return True
