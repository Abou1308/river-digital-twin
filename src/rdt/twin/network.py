"""Representation model of the river network: stations, reaches and what is known about them.

This is the static structure. The dynamic state (latest readings, estimates, alerts) lives in
`rdt.twin.state`, so the dashboard can read it without depending on the twin's internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rdt.config import Config
from rdt.data.schema import MODELLED_STATIONS, SensorInfo


@dataclass(frozen=True)
class Station:
    id: str
    river: str
    lat: float
    lon: float
    role: str                                   # "modelled" (soft sensor runs here) | "downstream"
    sensor_ids: tuple[str, ...] = field(default=())

    @property
    def is_modelled(self) -> bool:
        return self.id in MODELLED_STATIONS and self.role == "modelled"


@dataclass(frozen=True)
class Reach:
    """A river stretch between two stations.

    TODO (representation model): decide what a reach carries, e.g. travel time, length, and the
    hydrological state / sediment and metal fluxes propagated from the upstream station
    (flux = discharge * SPM concentration * metal content). Downstream stations (Keizersveer,
    Kampen; ~quarterly lab samples) can validate that propagation.
    """

    id: str
    river: str
    upstream: str
    downstream: str


@dataclass(frozen=True)
class RiverNetwork:
    stations: dict[str, Station]
    reaches: tuple[Reach, ...]

    @classmethod
    def from_config(cls, cfg: Config, catalog: dict[str, SensorInfo]) -> RiverNetwork:
        stations = {
            sid: Station(
                id=sid,
                river=s["river"],
                lat=float(s["lat"]),
                lon=float(s["lon"]),
                role=s.get("role", "downstream"),
                sensor_ids=tuple(k for k, info in catalog.items() if info.station == sid),
            )
            for sid, s in cfg.stations.items()
        }
        reaches = tuple(Reach(**r) for r in cfg.reaches)
        for r in reaches:
            missing = {r.upstream, r.downstream} - stations.keys()
            if missing:
                raise ValueError(f"Reach {r.id} references unknown stations: {missing}")
        return cls(stations=stations, reaches=reaches)

    def modelled_stations(self) -> list[Station]:
        return [s for s in self.stations.values() if s.is_modelled]
