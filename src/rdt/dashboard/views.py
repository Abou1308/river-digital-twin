"""Pure figure builders: data in, plotly Figure out. No Dash, no state - easy to test and reuse in notebooks."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go

from rdt.messaging.messages import SensorMessage
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


def network_map(network: RiverNetwork, station_class: dict[str, str | None]) -> go.Figure:
    """Stations as markers coloured by most likely class; reaches as lines.

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
                     "name": f"{int(level * 100)}% interval", "hoverinfo": "skip"})
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


def sensor_timeseries(series: dict[str, list[SensorMessage]], now: datetime | None, days: int = 60) -> Fig:
    """Recent automatic-sensor data (e.g. discharge, turbidity, SPM), one row per series, shared time axis."""
    names = list(series) or [""]
    n = len(names)
    start = (now - timedelta(days=days)) if now else None
    gap = 0.06
    h = (1 - gap * (n - 1)) / n
    data, layout = [], {"height": 120 + 160 * n, "margin": {"l": 50, "r": 10, "t": 30, "b": 30},
                        "uirevision": "sensors", "showlegend": False, "annotations": []}
    for i, name in enumerate(names):
        k = "" if i == 0 else str(i + 1)
        msgs = [m for m in series.get(name, []) if start is None or m.time_utc >= start]
        data.append({"type": "scattergl", "x": [m.time_utc for m in msgs], "y": [m.value for m in msgs],
                     "mode": "lines", "name": name, "xaxis": f"x{k}", "yaxis": f"y{k}"})
        top = 1 - i * (h + gap)
        layout[f"yaxis{k}"] = {"domain": [top - h, top], "anchor": f"x{k}"}
        layout[f"xaxis{k}"] = {"anchor": f"y{k}", "matches": "x" if i else None, "showticklabels": i == n - 1}
        layout["annotations"].append({"text": name, "xref": "paper", "yref": "paper", "x": 0.5, "y": top,
                                      "xanchor": "center", "yanchor": "bottom", "showarrow": False})
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
            "verdict": "about equal" if abs(et - er) < 1 else "twin better" if et < er else "reference better",
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
                      margin={"l": 50, "r": 10, "t": 30, "b": 30}, legend={"orientation": "h", "y": 1.15},
                      uirevision=f"run-{metal}")
    fig.update_yaxes(rangemode="tozero")
    return fig
