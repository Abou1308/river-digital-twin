"""Downstream check: lab results at the end of a reach against what the upstream estimate implies.

Keizersveer (Maas) and Kampen (IJssel) have quarterly lab samples of metals in suspended matter. Suspended
matter sampled there left the upstream station about one travel time earlier, so the expected content is

    ln(downstream) = ln(upstream estimate at t - travel) + f

where f is a learned factor per reach and metal: the median log ratio, over the training period, between
each downstream lab sample and the upstream lab sample nearest to t - travel (within `window_days`). It
absorbs tributary dilution (about a third of the Maas flow at Keizersveer comes from tributaries) and
sorting along the reach; the research found Keizersveer contents generally lower than Eijsden's.

The spread of the expectation combines the upstream estimate's variance and the spread of the training
log ratios (robust: 1.4826 x MAD). The ratio spread already contains upstream sample noise, so this is
somewhat too wide rather than too narrow.

Each downstream result is recorded as a `LabComparison` for the downstream station (the same record the
modelled stations use), and a result outside the 95% range is a "downstream_mismatch" alert.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from rdt.data.schema import SensorInfo, SensorKind
from rdt.messaging.messages import SensorMessage
from rdt.softsensor.estimate import MetalEstimate
from rdt.state import Alert, LabComparison, StateStore
from rdt.twin.components import TwinComponent
from rdt.twin.network import Reach

MIN_PAIRS = 8
_Z90, _Z95 = 1.6448536269514722, 1.959963984540054
_MATCH = pd.Timedelta(days=1)          # an upstream estimate this close to t - travel counts


@dataclass(frozen=True)
class DownstreamFactor:
    log_ratio: float       # median ln(downstream / upstream)
    sd: float              # robust sd of the log ratios
    n: int                 # training pairs


def fit_factors(readings: pd.DataFrame, reach: Reach, travel_days: float, metals: Sequence[str],
                window_days: float = 14.0) -> dict[str, DownstreamFactor]:
    """Factors per metal from training readings (long format: sensor_id, time_utc, value, censored)."""
    travel, window = pd.Timedelta(days=travel_days), pd.Timedelta(days=window_days)
    out = {}
    for m in metals:
        down = readings[(readings["sensor_id"] == f"{reach.downstream}_spm_{m}") & ~readings["censored"].astype(bool)]
        up = readings[(readings["sensor_id"] == f"{reach.upstream}_spm_{m}")
                      & ~readings["censored"].astype(bool)].sort_values("time_utc")
        up_t, up_v = pd.DatetimeIndex(up["time_utc"]), up["value"].to_numpy(float)
        ratios = []
        for t, v in zip(down["time_utc"], down["value"]):
            if not up_t.size or v <= 0:
                continue
            target = t - travel
            i = int(np.argmin(np.abs(up_t - target)))
            if abs(up_t[i] - target) <= window and up_v[i] > 0:
                ratios.append(math.log(v) - math.log(up_v[i]))
        if len(ratios) < MIN_PAIRS:
            raise ValueError(f"Only {len(ratios)} training pairs for {m} on {reach.id}; cannot fit its factor.")
        r = np.array(ratios)
        med = float(np.median(r))
        out[m] = DownstreamFactor(med, float(1.4826 * np.median(np.abs(r - med))), len(r))
    return out


class DownstreamCheck(TwinComponent):
    def __init__(self, reaches: Sequence[Reach], travel_days: Mapping[str, float],
                 factors: Mapping[str, Mapping[str, DownstreamFactor]]) -> None:
        self._by_downstream = {r.downstream: r for r in reaches}
        self.travel = {r.id: pd.Timedelta(days=travel_days[r.id]) for r in reaches}
        self.factors = {rid: dict(f) for rid, f in factors.items()}
        # Downstream lab results arrive after the lab turnaround; keep upstream estimates long enough.
        self._keep = max(self.travel.values()) + pd.Timedelta(days=30)
        self._history: dict[str, deque[MetalEstimate]] = {r.upstream: deque() for r in reaches}

    def on_estimate(self, est: MetalEstimate, store: StateStore) -> None:
        history = self._history.get(est.station)
        if history is None:
            return
        history.append(est)
        while history and pd.Timestamp(history[0].time_utc) < pd.Timestamp(est.time_utc) - self._keep:
            history.popleft()

    def on_reading(self, info: SensorInfo, msg: SensorMessage, store: StateStore) -> None:
        reach = self._by_downstream.get(info.station)
        if reach is None or info.kind is not SensorKind.SPM_METAL or msg.censored or not msg.value > 0:
            return
        factor = self.factors[reach.id].get(info.metal)
        upstream = self._upstream_estimate(reach, pd.Timestamp(msg.time_utc))
        if factor is None or upstream is None or info.metal not in upstream.metals:
            return
        j = upstream.metals.index(info.metal)
        mean = upstream.mean_log[j] + factor.log_ratio
        sd = math.sqrt(upstream.cov_log[j, j] + factor.sd**2)
        store.record_comparison(LabComparison(
            time_utc=msg.time_utc, station=info.station, metal=info.metal, observed=msg.value,
            predicted_median=math.exp(mean), lower=math.exp(mean - _Z90 * sd), upper=math.exp(mean + _Z90 * sd),
            censored=False))
        if abs(math.log(msg.value) - mean) > _Z95 * sd:
            store.record_alert(Alert(
                msg.arrival_utc, info.station, "downstream_mismatch",
                f"{info.station} {info.metal}: lab {msg.value:.3g} mg/kg, expected {math.exp(mean):.3g} "
                f"(95% {math.exp(mean - _Z95 * sd):.3g}-{math.exp(mean + _Z95 * sd):.3g}) from {reach.upstream} "
                f"{self.travel[reach.id].days} days earlier", metal=info.metal))

    def _upstream_estimate(self, reach: Reach, sample_time: pd.Timestamp) -> MetalEstimate | None:
        target = sample_time - self.travel[reach.id]
        best = min(self._history[reach.upstream], default=None,
                   key=lambda e: abs(pd.Timestamp(e.time_utc) - target))
        if best is None or abs(pd.Timestamp(best.time_utc) - target) > _MATCH:
            return None
        return best
