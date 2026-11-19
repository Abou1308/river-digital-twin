"""Optional twin components: parts of the representation model that react to readings and estimates.

The twin calls every configured component (config `twin.components`) on each reading, from every station,
and on each new soft-sensor estimate. A component keeps its own state and writes it to the store: the
current state of its entities as `EntityState` records, and events as `Alert`s. Components only add to
the twin; removing one from the config switches it off without other changes.
"""

from __future__ import annotations

from rdt.data.schema import SensorInfo
from rdt.messaging.messages import SensorMessage
from rdt.softsensor.estimate import MetalEstimate
from rdt.state import StateStore


class TwinComponent:
    """Base class; override the hooks a component needs (both do nothing by default)."""

    def on_reading(self, info: SensorInfo, msg: SensorMessage, store: StateStore) -> None:  # noqa: B027
        """Called for every reading, after it is recorded; `msg.arrival_utc` is the current time."""

    def on_estimate(self, est: MetalEstimate, store: StateStore) -> None:  # noqa: B027
        """Called for every new estimate of a modelled station, after it is recorded."""
