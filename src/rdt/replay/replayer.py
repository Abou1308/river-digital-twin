"""Streams readings onto the broker in time order, paced by a `SimClock`."""

from __future__ import annotations

import logging
import threading

import pandas as pd

from rdt.data.schema import SensorInfo
from rdt.messaging.broker import Broker
from rdt.messaging.messages import SensorMessage
from rdt.replay.clock import SimClock

log = logging.getLogger(__name__)


class Replayer:
    """Publishes each row of `readings` to its sensor's topic when the simulated clock reaches it.

    `readings` must already be restricted to the replay window (and to the tiers being streamed):
    the replayer does no filtering, which keeps the leakage boundary in one place (the caller).
    If it has an `available_utc` column (lab results published after the lab turnaround), messages
    are published at that time and `readings` must be sorted by it; otherwise at `time_utc`.
    """

    def __init__(
        self,
        readings: pd.DataFrame,
        catalog: dict[str, SensorInfo],
        broker: Broker,
        clock: SimClock,
    ) -> None:
        self.readings = readings
        self.catalog = catalog
        self.broker = broker
        self.clock = clock
        self.stop_event = threading.Event()
        self.published = 0
        self.finished = threading.Event()
        self.error: Exception | None = None      # set if publishing (i.e. a subscriber) raised

    def run(self) -> int:
        """Replay all readings (blocking). Returns the number of messages published."""
        r = self.readings
        avail = r["available_utc"] if "available_utc" in r.columns else r["time_utc"]
        rows = zip(r["time_utc"], avail, r["sensor_id"], r["value"], r["censored"], r["qc"])
        try:
            for t, at, sid, value, censored, qc in rows:
                if not self.clock.wait_until(at, self.stop_event):
                    break
                info = self.catalog[sid]
                msg = SensorMessage(
                    sensor_id=sid,
                    time_utc=t.to_pydatetime(),
                    value=float(value),
                    censored=bool(censored),
                    qc=None if pd.isna(qc) else str(qc),
                    available_utc=None if at == t else at.to_pydatetime(),
                )
                self.broker.publish(info.topic, msg)
                self.published += 1
        except Exception as e:
            self.error = e
            log.error("Replay stopped after %d messages: %s", self.published, e)
            raise
        else:
            log.info("Replay finished: %d messages published", self.published)
        finally:
            self.finished.set()
        return self.published

    def start(self) -> threading.Thread:
        """Run the replay in a daemon thread."""
        thread = threading.Thread(target=self.run, name="replayer", daemon=True)
        thread.start()
        return thread

    def stop(self) -> None:
        self.stop_event.set()
