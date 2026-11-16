"""The single message type that travels over the broker: one sensor reading."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class SensorMessage:
    sensor_id: str
    time_utc: datetime          # timezone-aware, UTC
    value: float
    censored: bool = False      # value is a detection limit, true value is below it
    qc: str | None = None       # Rijkswaterstaat quality code, kept as-is
    # When the reading was published. Sensor readings: = time_utc (None). Lab results: time_utc is the
    # *sample* time, and the result is only published after the lab turnaround (available_utc).
    available_utc: datetime | None = None

    @property
    def arrival_utc(self) -> datetime:
        return self.available_utc if self.available_utc is not None else self.time_utc

    def to_json(self) -> str:
        return json.dumps(
            {
                "sensor_id": self.sensor_id,
                "time_utc": self.time_utc.isoformat(),
                "value": None if math.isnan(self.value) else self.value,
                "censored": self.censored,
                "qc": self.qc,
                "available_utc": None if self.available_utc is None else self.available_utc.isoformat(),
            }
        )

    @classmethod
    def from_json(cls, payload: str | bytes) -> SensorMessage:
        d = json.loads(payload)
        return cls(
            sensor_id=d["sensor_id"],
            time_utc=datetime.fromisoformat(d["time_utc"]),
            value=float("nan") if d["value"] is None else float(d["value"]),
            censored=bool(d.get("censored", False)),
            qc=d.get("qc"),
            available_utc=None if d.get("available_utc") is None else datetime.fromisoformat(d["available_utc"]),
        )
