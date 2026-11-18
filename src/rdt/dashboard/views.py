"""Pure figure builders: data in, plotly Figure out. No Dash, no state - easy to test and reuse in notebooks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, time

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from rdt.messaging.messages import SensorMessage
from rdt.quality import sample_class_probabilities
from rdt.softsensor.estimate import MetalEstimate
from rdt.state import Alert, LabComparison
from rdt.twin.network import RiverNetwork

# Quality-class colours (clean -> heavily contaminated); grey = no estimate yet.
CLASS_COLOURS = {
    "clean": "#2e7d32",
    "slightly contaminated": "#f9a825",
    "moderately contaminated": "#ef6c00",
    "heavily contaminated": "#c62828",
    None: "#9e9e9e",
}
# Text on a class colour: dark on the light yellow, white on the others, for legible contrast.
CLASS_TEXT_COLOURS = {cls: "#14273a" if cls == "slightly contaminated" else "#ffffff" for cls in CLASS_COLOURS}


def network_map(network: RiverNetwork, station_class: dict[str, str | None]) -> go.Figure:
    """Stations as markers coloured by most likely class; reaches as lines.

    Not shown on the dashboard yet: it only becomes useful once it can show the contamination carried
    along the river (estimated metal levels / fluxes per reach).

    TODO: draw reaches along the actual river geometry (e.g. from a GeoJSON) instead of straight lines,
    and colour reaches by their predicted class once the representation model propagates fluxes.
    """
    fig = go.Figure()
    for r in network.reaches:
        up, down = network.stations[r.upstream], network.stations[r.downstream]
        fig.add_trace(
            go.Scattermap(
                lat=[up.lat, down.lat], lon=[up.lon, down.lon], mode="lines",
                line={"width": 3, "color": "#5c8fbf"}, name=r.id, hoverinfo="name", showlegend=False,
            )
        )
    stations = list(network.stations.values())
    classes = [station_class.get(s.id) for s in stations]
    fig.add_trace(
        go.Scattermap(
            lat=[s.lat for s in stations],
            lon=[s.lon for s in stations],
            mode="markers+text",
            marker={"size": [16 if s.is_modelled else 10 for s in stations],
                    "color": [CLASS_COLOURS.get(c, CLASS_COLOURS[None]) for c in classes]},
            text=[s.id.capitalize() for s in stations],
            textposition="top right",
            customdata=[s.id for s in stations],
            hovertext=[f"{s.id} ({s.river})<br>class: {c or 'n/a'}" for s, c in zip(stations, classes)],
            hoverinfo="text",
            showlegend=False,
        )
    )
    fig.update_layout(
        map={"style": "open-street-map", "center": {"lat": 51.7, "lon": 5.6}, "zoom": 6.3},
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )
    return fig


class EstimateSeries:
    """Median and interval per metal for a growing list of estimates, computed once per estimate.

    The dashboard redraws every second; recomputing thousands of intervals each time slowed the replay
    (same process). Estimates are append-only, so only new ones are processed."""

    def __init__(self, level: float = 0.9) -> None:
        self.level = level
        self._n: dict[str, int] = {}
        self._rows: dict[str, dict[str, list[tuple]]] = {}

    def get(self, key: str, estimates: list[MetalEstimate], metal: str) -> list[tuple]:
        """[(ISO time, median, lower, upper)] for `metal`; `key` identifies the estimate stream."""
        n = self._n.get(key, 0)
        if len(estimates) < n:                       # stream restarted / truncated: rebuild
            n, self._rows[key] = 0, {}
        rows = self._rows.setdefault(key, {})
        for e in estimates[n:]:
            med = e.median()
            lo, hi = e.interval(self.level)
            for m in e.metals:
                # time as an ISO string, formatted once: much faster for Dash to serialise every second
                rows.setdefault(m, []).append((e.time_utc.isoformat(), float(med[m]), float(lo[m]), float(hi[m])))
        self._n[key] = len(estimates)
        return rows.get(metal, [])


# The two big charts are returned as plain figure dicts (which Dash accepts): building go.Figure
# objects validates every data point, which took ~0.1 s per trace with a few years of history and,
# with the dashboard redrawing every second in the same process, slowed the replay down.
Fig = dict


def metal_timeseries(
    metal: str,
    estimates: list[tuple],
    lab: list[SensorMessage],
    limits: pd.DataFrame | None = None,
    level: float = 0.9,
    reference: list[tuple] | None = None,
) -> Fig:
    """Soft-sensor median with `level` band, lab results as points, standard-soil class limits as lines,
    and (optional) the reference model (last lab value known) as a dashed line.
    `estimates` / `reference`: [(time, median, lower, upper)] rows from `EstimateSeries.get`."""
    data: list[dict] = []
    if estimates:
        t, med, lo, hi = (list(c) for c in zip(*estimates))
        data.append({"type": "scatter", "x": t + t[::-1], "y": hi + lo[::-1], "fill": "toself",
                     "line": {"width": 0}, "fillcolor": "rgba(92,143,191,0.25)",
                     "name": f"{int(level * 100)}% range for a lab result", "hoverinfo": "skip"})
        data.append({"type": "scatter", "x": t, "y": med, "mode": "lines", "line": {"color": "#2a5d8f"},
                     "name": "twin estimate"})
    if reference:
        data.append({"type": "scatter", "x": [r[0] for r in reference], "y": [r[1] for r in reference],
                     "mode": "lines", "line": {"color": "#888", "dash": "dash", "shape": "hv"},
                     "name": "last lab value known (no twin)"})
    if lab:
        data.append({"type": "scatter", "x": [m.time_utc for m in lab], "y": [m.value for m in lab],
                     "mode": "markers", "name": "lab",
                     "marker": {"color": ["#ffffff" if m.censored else "#111111" for m in lab],
                                "line": {"color": "#111111", "width": 1}, "size": 7}})
    shapes, notes = [], []
    if limits is not None:
        rows = limits[(limits["parameter"].str.lower() == metal) & limits["upper_inclusive"].notna()]
        for _, r in rows.iterrows():
            y = float(r["upper_inclusive"])
            shapes.append({"type": "line", "xref": "paper", "x0": 0, "x1": 1, "y0": y, "y1": y,
                           "line": {"color": "#888", "dash": "dot"}})
            notes.append({"xref": "paper", "x": 0, "y": y, "text": f"{r['class_en']} (std. soil)",
                          "showarrow": False, "xanchor": "left", "yanchor": "bottom"})
    layout = {"yaxis": {"title": {"text": f"{metal.upper()} in SPM (mg/kg dw)"}},
              "margin": {"l": 50, "r": 10, "t": 30, "b": 30}, "legend": {"orientation": "h", "y": 1.1},
              "uirevision": metal, "shapes": shapes, "annotations": notes}
    return {"data": data, "layout": layout}


class DailyStats:
    """Per-day sum, count and max of one sensor stream, updated incrementally: each call re-reads only
    the readings from the start of the last (still incomplete) day onward, so the cost stays small
    however long the history is. Readings that arrive later for an already completed day are ignored."""

    def __init__(self) -> None:
        self._bins: dict[str, dict[date, list[float]]] = {}
        self._last: dict[str, datetime] = {}

    def get(self, key: str, fetch: Callable[[datetime | None], list[SensorMessage]],
            keep: Callable[[float], bool] = lambda v: True) -> dict[date, list[float]]:
        """`fetch(since)` returns readings with time >= since (all if None); `keep` filters values
        (e.g. the twin's arrival rules). Returns {day: [sum, count, max]}."""
        bins = self._bins.setdefault(key, {})
        since = self._last.get(key)
        msgs = fetch(since)
        if msgs:
            fresh: dict[date, list[float]] = {}
            for m in msgs:
                if keep(m.value):
                    b = fresh.setdefault(m.time_utc.date(), [0.0, 0, float("-inf")])
                    b[0] += m.value
                    b[1] += 1
                    b[2] = max(b[2], m.value)
            if since is not None:
                bins.pop(since.date(), None)
            bins.update(fresh)
            last = msgs[-1].time_utc
            self._last[key] = datetime.combine(last.date(), time(), tzinfo=last.tzinfo)
        return bins


def window_series(bins: dict[date, list[float]], stat: str, days: int) -> tuple[list[str], list[float]]:
    """Trailing `days`-day mean or max from daily stats, at day resolution, plotted at the end of each
    window (midnight after the last day) - i.e. the value the soft sensor would see at that moment.
    Same definition as the model inputs (rdt/features/windows.py), rounded to whole days."""
    if not bins:
        return [], []
    df = pd.DataFrame.from_dict(bins, orient="index", columns=["sum", "n", "max"]).sort_index()
    df.index = pd.DatetimeIndex(df.index)
    df = df.asfreq("D")                                  # missing days -> NaN, so windows stay in days
    r = df.rolling(days, min_periods=1)
    y = r["max"].max() if stat == "max" else r["sum"].sum() / r["n"].sum().replace(0, float("nan"))
    y = y.dropna()
    x = (y.index + pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00:00")
    return list(x), [float(v) for v in y]


# --- look -------------------------------------------------------------------------------------
FONT = "'IBM Plex Mono', ui-monospace, Menlo, Consolas, monospace"
AXIS = {"showline": True, "mirror": True, "linecolor": "#7d93a8", "linewidth": 1, "gridcolor": "#e3ebf3",
        "zeroline": False, "ticks": "outside", "tickcolor": "#7d93a8"}
THEME = {"font": {"family": FONT, "size": 12, "color": "#14273a"}, "paper_bgcolor": "#ffffff",
         "plot_bgcolor": "#f7fafd", "hoverlabel": {"font": {"family": FONT}}}
# one colour per sensor kind, distinct from the metal estimate (blue) and lab points (black)
SENSOR_COLOURS = {"discharge": "#0e8f8f", "turbidity": "#c97a14", "spm_daily": "#8b4fa8"}


def with_sensor_panels(metal_fig: Fig, panels: list[tuple[str, str, list, list]],
                       metal_share: float = 0.46) -> Fig:
    """Stack sensor panels [(sensor kind, title, x, y)] under the metal chart on one shared time axis,
    so a flood or turbidity peak lines up with the metal estimate above it; zooming one zooms all.
    Every panel is boxed and titled, with a gap between panels."""
    n = len(panels)
    gap = 0.075
    h = (1 - metal_share - gap * n) / max(n, 1)
    data = list(metal_fig["data"])
    layout = {**metal_fig["layout"], **THEME}
    ytitle = layout.get("yaxis", {}).get("title", {}).get("text", "")
    layout["yaxis"] = {**AXIS, "domain": [1 - metal_share, 1]}
    layout["xaxis"] = {**AXIS, "anchor": "y", "showticklabels": n == 0}
    notes = list(layout.get("annotations", []))
    notes.append({"text": f"<b>{ytitle}</b>", "xref": "paper", "yref": "paper", "x": 0, "y": 1,
                  "xanchor": "left", "yanchor": "bottom", "yshift": 4, "showarrow": False,
                  "font": {"size": 13, "color": "#2a5d8f"}})
    for i, (kind, title, x, y) in enumerate(panels, start=2):
        top = 1 - metal_share - gap - (i - 2) * (h + gap)
        colour = SENSOR_COLOURS.get(kind, "#555")
        data.append({"type": "scatter", "x": x, "y": y, "mode": "lines", "line": {"color": colour, "width": 1.6},
                     "name": title, "xaxis": f"x{i}", "yaxis": f"y{i}", "showlegend": False,
                     "hovertemplate": "%{y:.4g}<extra></extra>"})
        layout[f"yaxis{i}"] = {**AXIS, "domain": [top - h, top], "anchor": f"x{i}"}
        layout[f"xaxis{i}"] = {**AXIS, "anchor": f"y{i}", "matches": "x", "showticklabels": i == n + 1}
        notes.append({"text": f"<b>{title}</b>", "xref": "paper", "yref": "paper", "x": 0, "y": top,
                      "xanchor": "left", "yanchor": "bottom", "yshift": 4, "showarrow": False,
                      "font": {"size": 13, "color": colour}})
    layout["annotations"] = notes
    layout["legend"] = {"orientation": "h", "y": 1, "yanchor": "bottom", "x": 1, "xanchor": "right",
                        "bgcolor": "rgba(0,0,0,0)"}
    layout["margin"] = {"l": 64, "r": 16, "t": 48, "b": 36}
    layout["hovermode"] = "x"
    return {"data": data, "layout": layout}


def alerts_rows(alerts: list[Alert]) -> list[dict[str, str]]:
    return [
        {"time": a.time_utc.strftime("%Y-%m-%d %H:%M"), "station": a.station, "metal": (a.metal or "").upper(),
         "kind": a.kind, "severity": a.severity, "message": a.message}
        for a in alerts
    ]


# --- live scoreboard: twin vs reference, over the lab results received so far ------------------

def _comparison_frame(comparisons: list[LabComparison]) -> pd.DataFrame:
    rows = [c for c in comparisons if c.reference_median is not None and not c.censored and c.observed > 0]
    df = pd.DataFrame([{"time": c.time_utc, "metal": c.metal, "obs": c.observed, "twin": c.predicted_median,
                        "ref": c.reference_median, "lo": c.lower, "hi": c.upper} for c in rows])
    if df.empty:
        return df
    df["err_twin"] = (df.twin - df.obs).abs() / df.obs * 100
    df["err_ref"] = (df.ref - df.obs).abs() / df.obs * 100
    df["closer"] = df.err_twin < df.err_ref
    df["inside"] = (df.obs >= df.lo) & (df.obs <= df.hi)
    return df.sort_values("time")


def scoreboard_rows(comparisons: list[LabComparison], metals: tuple[str, ...]) -> list[dict[str, str]]:
    """Per metal: each lab result compared with what the twin and the reference predicted for the sample
    time, before the result was known."""
    df = _comparison_frame(comparisons)
    out = []
    for m in metals:
        g = df[df.metal == m] if not df.empty else df
        if g.empty:
            out.append({"metal": m.upper(), "lab results": "0"})
            continue
        et, er = g.err_twin.median(), g.err_ref.median()
        out.append({
            "metal": m.upper(),
            "lab results": str(len(g)),
            "twin error (median)": f"{et:.1f}%",
            "last-lab-value error (median)": f"{er:.1f}%",
            "twin closer": f"{g.closer.mean() * 100:.0f}% of results",
            "inside twin's 90% range": f"{g.inside.mean() * 100:.0f}%",
            # running tally, not a significance test (see analysis/04_soft_sensor_evaluation.py)
            "verdict": "about equal" if abs(et - er) < 1 else "twin better" if et < er else "last lab value better",
        })
    return out


def running_error(comparisons: list[LabComparison], metal: str) -> go.Figure:
    """Median % error over all lab results received so far, twin vs reference, as results come in."""
    df = _comparison_frame(comparisons)
    fig = go.Figure()
    if not df.empty and (df.metal == metal).any():
        g = df[df.metal == metal]
        fig.add_trace(go.Scatter(x=g.time, y=g.err_twin.expanding().median(), mode="lines",
                                 line={"color": "#2a5d8f"}, name="twin"))
        fig.add_trace(go.Scatter(x=g.time, y=g.err_ref.expanding().median(), mode="lines",
                                 line={"color": "#888", "dash": "dash"}, name="last lab value known (no twin)"))
    fig.update_layout(yaxis_title=f"{metal.upper()}: median % error so far",
                      margin={"l": 60, "r": 16, "t": 30, "b": 36}, legend={"orientation": "h", "y": 1.15},
                      uirevision=f"run-{metal}", **THEME)
    fig.update_xaxes(**AXIS)
    fig.update_yaxes(**AXIS, rangemode="tozero")
    return fig


@dataclass(frozen=True)
class ClassSummary:
    """Current class probabilities at a station, or the reason there are none."""

    probabilities: dict[str, float] | None
    reason: str = ""
    lutum_pct: float | None = None
    om_pct: float | None = None
    soil_sample_time: datetime | None = None

    @property
    def most_likely(self) -> str | None:
        return max(self.probabilities, key=self.probabilities.get) if self.probabilities else None


def class_summary(
    estimate: MetalEstimate | None,
    lutum: SensorMessage | None,
    corg: SensorMessage | None,
    limits: pd.DataFrame,
    soil_correction: pd.DataFrame,
    om_per_corg: float,
) -> ClassSummary:
    """Class probabilities of the latest estimate, corrected with the last lutum / organic carbon received."""
    if estimate is None:
        return ClassSummary(None, "no estimate yet")
    if lutum is None or corg is None:
        return ClassSummary(None, "no class: lutum / organic carbon unknown")
    om = om_per_corg * corg.value
    # A fixed seed gives the same numbers for the same estimate, so the panel does not flicker on refresh.
    probs = sample_class_probabilities(estimate, limits, soil_correction, lutum.value, om,
                                       rng=np.random.default_rng(0))
    return ClassSummary(probs, lutum_pct=lutum.value, om_pct=om,
                        soil_sample_time=min(lutum.time_utc, corg.time_utc))
