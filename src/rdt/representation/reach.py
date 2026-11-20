"""Reaches: sediment and metal flux from the upstream station, and the fresh deposit they build up.

For each reach, at every new estimate of its upstream station (every 6 h):

  discharge into the reach  = share(Q) x Q,  Q = 24-h mean discharge at the upstream station
  SPM                       = the latest daily composite (at most 48 h old), else from the 24-h mean
                              turbidity via the training turbidity-SPM relation (turbidity_spm.py)
  sediment load (t/day)     = discharge x SPM x 0.0864           (m3/s x mg/l -> t/day)
  metal flux (kg/day)       = sediment load x content x 1e-3     (content: soft-sensor median, mg/kg)

`share(Q)` is the fraction of the upstream discharge that enters the reach, interpolated in a table of
discharge -> fraction (the Rhine splits at the Pannerdensche Kop and IJsselkop, and the IJssel share
depends on discharge; suspended sediment divides roughly like the water). The Maas reach has share 1;
its tributaries are not modelled.

The load and the estimated content enter the reach's fresh deposit (deposit.py) after the reach's travel
time. The deposit's class uses the latest lab lutum and organic carbon of the upstream station, as the
station's own class does.

Entity "reach" (id = reach id) holds the current flux and the deposit; a change of the deposit's most
likely class is a "class_change" alert (warning if worse, info if better).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from rdt.data.cleaning import passes_arrival_rules
from rdt.quality import QUALITY_CLASSES, draws_class_probabilities
from rdt.representation.deposit import Deposit
from rdt.representation.turbidity_spm import MIN_READINGS, TurbiditySpmRelation
from rdt.softsensor.estimate import MetalEstimate
from rdt.state import Alert, EntityState, StateStore
from rdt.twin.components import TwinComponent
from rdt.twin.network import Reach

LOAD_FACTOR = 0.0864       # m3/s x mg/l -> tonnes/day
_DAY = pd.Timedelta(hours=24)


@dataclass(frozen=True)
class ReachSettings:
    travel_days: float                 # upstream station -> downstream end
    share_q: tuple[float, ...]         # upstream discharge (m3/s) ...
    share: tuple[float, ...]           # ... and the fraction of it that enters the reach

    def fraction(self, q: float) -> float:
        """Share of the upstream discharge entering the reach; constant beyond the table's ends."""
        return float(np.interp(q, self.share_q, self.share))


def sediment_load(discharge_m3s: float, spm_mg_l: float) -> float:
    """Suspended sediment load in tonnes per day."""
    return discharge_m3s * spm_mg_l * LOAD_FACTOR


def _mean_valid(messages: Sequence, kind: str, min_n: int) -> float | None:
    values = [m.value for m in messages if passes_arrival_rules(kind, m.value)]
    return float(np.mean(values)) if len(values) >= min_n else None


@dataclass
class _Reach:
    reach: Reach
    settings: ReachSettings
    deposit: Deposit
    queue: deque          # (due time, load, dt_days, mean_log, cov_log) waiting for the travel time
    last_class: str | None = None


class ReachState(TwinComponent):
    def __init__(
        self,
        reaches: Sequence[Reach],
        settings: Mapping[str, ReachSettings],
        metals: Sequence[str],
        relations: Mapping[str, TurbiditySpmRelation],
        limits: pd.DataFrame,
        soil_correction: pd.DataFrame,
        om_per_corg: float,
        residence_days: float = 30.0,
        n_draws: int = 2000,
    ) -> None:
        missing = [r.id for r in reaches if r.id not in settings]
        if missing:
            raise ValueError(f"No reach settings for {missing} (config representation.reach.reaches).")
        self.relations = dict(relations)
        self.limits, self.soil_correction, self.om_per_corg = limits, soil_correction, om_per_corg
        self.metals = tuple(metals)
        self._reaches = [
            _Reach(r, settings[r.id], Deposit(residence_days, len(self.metals), n_draws), deque())
            for r in reaches
        ]
        self._last_estimate: dict[str, pd.Timestamp] = {}

    def on_estimate(self, est: MetalEstimate, store: StateStore) -> None:
        st, t = est.station, pd.Timestamp(est.time_utc)
        previous = self._last_estimate.get(st)
        self._last_estimate[st] = t
        for r in self._reaches:
            if r.reach.upstream != st:
                continue
            if tuple(est.metals) != self.metals:
                raise ValueError(f"Estimate metals {est.metals} differ from the reach metals {self.metals}.")
            self._step(r, est, t, previous, store)

    def _inputs(self, st: str, t: pd.Timestamp, store: StateStore) -> tuple[float | None, float | None, str]:
        """24-h mean discharge, SPM (mg/l) and where the SPM came from."""
        q = _mean_valid(store.readings(f"{st}_discharge", since=t - _DAY), "discharge", MIN_READINGS)
        composites = [m for m in store.readings(f"{st}_spm_daily", since=t - 2 * _DAY)
                      if not m.censored and passes_arrival_rules("spm_daily", m.value)]
        if composites:
            return q, composites[-1].value, "daily composite"
        turbidity = _mean_valid(store.readings(f"{st}_turbidity", since=t - _DAY), "turbidity", MIN_READINGS)
        if turbidity is not None and st in self.relations:
            return q, float(np.exp(self.relations[st].log_spm(turbidity))), "turbidity"
        return q, None, "none"

    def _step(self, r: _Reach, est: MetalEstimate, t: pd.Timestamp, previous: pd.Timestamp | None,
              store: StateStore) -> None:
        st = r.reach.upstream
        q, spm, source = self._inputs(st, t, store)
        values: dict = {"spm_source": source}
        if q is not None and spm is not None and previous is not None:
            share = r.settings.fraction(q)
            load = sediment_load(share * q, spm)
            median = dict(zip(est.metals, np.exp(est.mean_log)))
            values |= {"share": share, "discharge_m3s": share * q, "spm_mg_l": spm, "sediment_load_t_per_d": load,
                       "metal_flux_kg_per_d": {m: load * c * 1e-3 for m, c in median.items()}}
            dt_days = (t - previous) / _DAY
            r.queue.append((t + pd.Timedelta(days=r.settings.travel_days), load, dt_days, est.mean_log, est.cov_log))
        while r.queue and r.queue[0][0] <= t:
            due, load, dt_days, mean_log, cov_log = r.queue.popleft()
            r.deposit.add(due, load, dt_days, mean_log, cov_log)
        if r.deposit.ready:
            values |= self._deposit_values(r, est.metals, st, store)
            self._class_event(r, values.get("most_likely_class"), est, store)
        store.record_entity(EntityState("reach", r.reach.id, est.time_utc, values))

    def _deposit_values(self, r: _Reach, metals: Sequence[str], st: str, store: StateStore) -> dict:
        draws = r.deposit.draws()
        out: dict = {"deposit_median": dict(zip(metals, np.median(draws, axis=0))),
                     "deposit_p05": dict(zip(metals, np.quantile(draws, 0.05, axis=0))),
                     "deposit_p95": dict(zip(metals, np.quantile(draws, 0.95, axis=0)))}
        lutum, corg = store.latest_reading(f"{st}_spm_lutum"), store.latest_reading(f"{st}_spm_corg")
        if lutum is None or corg is None:
            out["class_reason"] = "no lab lutum / organic carbon yet"
            return out
        probs = draws_class_probabilities(draws, metals, self.limits, self.soil_correction, lutum.value,
                                          self.om_per_corg * corg.value)
        out |= {"class_probabilities": probs, "most_likely_class": max(probs, key=probs.get)}
        return out

    def _class_event(self, r: _Reach, new: str | None, est: MetalEstimate, store: StateStore) -> None:
        old, r.last_class = r.last_class, new or r.last_class
        if old is None or new is None or new == old:
            return
        worse = QUALITY_CLASSES.index(new) > QUALITY_CLASSES.index(old)
        store.record_alert(Alert(est.time_utc, r.reach.downstream, "class_change",
                                 f"{r.reach.id}: fresh deposit most likely {new} (was {old})",
                                 severity="warning" if worse else "info"))
