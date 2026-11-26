"""Dash app. Reads only from the `StateStore` (never from the twin directly), polling on an interval.

What the dashboard shows (kept here rather than as text on the page):

River network map: modelled stations coloured by the most likely class of their suspended sediment (as in
the quality-class panel), reaches coloured by the reported class of their fresh deposit
(rdt/representation/reach.py, with hysteresis), downstream stations in grey. Clicking a reach or station
shows that reach's current state beside the map: the upstream regime, discharge entering the reach, the
suspended matter and sediment load, the metal flux, and the deposit's content (median and 90% range of
the joint draws) and class probabilities.

Main chart, top panel: the twin's estimate of the selected metal in suspended sediment at the selected
station (median, solid line), lab results (points; white = below detection limit), the last lab value
that was known at each moment (dashed: what you would have without the twin; it steps when a result
arrives, `softsensor.lab_delay_d` after sampling) and the standard-soil class limits (dotted).
The shaded band is the 90% range for a lab result: where a sample taken at that moment is expected to
fall 9 times out of 10, given the sensor data and the lab results received so far. On the log scale it
is mean +- 1.645 sd, with variance P (level uncertainty) + Q * days since the last result (drift) +
R (sample-to-sample noise), widened by a per-metal factor calibrated on the training period
(see rdt/softsensor/kalman.py). It widens between lab results and narrows when one arrives.

Main chart, lower panels: the sensor inputs on the same time axis (zooming one zooms all). Each panel
has a selector for the window statistic shown (e.g. 24-h / 7-day / 30-day mean, 30-day max), computed
from the readings as the twin receives them (arrival rules applied) at day resolution and plotted at
the end of the window. Options marked "model input" are the ones the soft sensor uses.

Accuracy table: each lab result, when it arrives, is compared with what the twin predicted for the
sample time and with the last lab value known then (both before the result was known).
Error = |prediction - lab| / lab; "about equal" = less than 1 percentage point apart. This is a running
tally, not a significance test: the statistical tests are in analysis/04_soft_sensor_evaluation.py.

Quality class, per station: the probability of each Rbk 2022 class for the latest estimate (worst class
over the five metals, from joint draws), with the class limits corrected to the last lutum and organic
carbon received (organic matter = `quality.om_per_corg` x organic carbon). The dot shows the most likely
class. Below it, the class forecast so far: each complete lab sample's class compared with the class
probabilities predicted for its sample time, for the twin and the last known lab value (Brier score:
0 is perfect, lower is better; accuracy of the most likely class).

Reuse line (station and reach panels): the probability that the sediment, if dredged, may be applied in
surface water onto a water bottom of each class (Bal 4.1272: material may go onto a bottom of its own
class or worse; heavily contaminated material not at all). It follows from the class probabilities, so
it rests on the five estimated metals only (rdt/quality/reuse.py).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from dash import Dash, Input, Output, dash_table, dcc, html

from rdt.dashboard import views
from rdt.monitoring.classes import score_class_comparisons
from rdt.quality import QUALITY_CLASSES, surface_water_probabilities
from rdt.data.cleaning import passes_arrival_rules
from rdt.state import ClassComparison, EntityState, StateStore
from rdt.twin.network import RiverNetwork

TITLE = "Digital Twin of Heavy-Metal Transport Via Suspended Sediments: In the Rhine & Maas"

# Sensor panels under the metal chart: kind -> (title, feature prefix, {option value: label}).
# Option values are "<stat>_<days>"; they mirror the model's window features (rdt/features/windows.py).
SENSOR_PANELS: dict[str, tuple[str, str, dict[str, str]]] = {
    "discharge": ("Discharge", "q", {"mean_1": "24-h mean", "mean_3": "3-day mean", "mean_7": "7-day mean",
                                     "mean_30": "30-day mean", "max_7": "7-day max", "max_30": "30-day max"}),
    "turbidity": ("Turbidity", "turb", {"mean_1": "24-h mean", "mean_3": "3-day mean", "mean_7": "7-day mean",
                                        "mean_30": "30-day mean", "max_7": "7-day max", "max_30": "30-day max"}),
    "spm_daily": ("Suspended matter", "spm", {"mean_1": "24-h sample", "mean_7": "7-day mean",
                                              "mean_30": "30-day mean", "max_30": "30-day max"}),
}
_DAYS_TO_WINDOW = {1: "24h", 3: "3d", 7: "7d", 30: "30d"}
_TABLE_CELL = {"textAlign": "left", "fontSize": "0.85em", "fontFamily": views.FONT, "padding": "6px 8px"}
_TABLE_HEADER = {"fontWeight": "600", "backgroundColor": "#e8f0f8", "borderBottom": "1px solid #7d93a8"}


def _feature_name(prefix: str, option: str) -> str:
    stat, days = option.split("_")
    return f"{prefix}_{stat}_{_DAYS_TO_WINDOW[int(days)]}"


def _control(label: str, component) -> html.Div:
    return html.Div(className="control", children=[html.Label(label), component])


def _class_bar(probabilities: dict[str, float]) -> html.Div:
    segments = []
    for cls in QUALITY_CLASSES:
        p = probabilities[cls]
        if p <= 0:
            continue
        segments.append(html.Div(f"{p:.0%}" if p >= 0.12 else "", className="class-seg", title=f"{cls}: {p:.1%}",
                                 style={"width": f"{p * 100:.2f}%", "background": views.CLASS_COLOURS[cls],
                                        "color": views.CLASS_TEXT_COLOURS[cls]}))
    return html.Div(segments, className="class-bar")


def _reuse_line(probabilities: dict[str, float]) -> html.Div:
    """Probability that the sediment, if dredged, may be applied in surface water, per class of the
    receiving water bottom (rdt/quality/reuse.py)."""
    p = surface_water_probabilities(probabilities)
    parts = ", ".join(f"onto a {bottom} water bottom {share:.0%}" for bottom, share in p.items())
    return html.Div(f"If dredged, applicable in surface water (Bal 4.1272; judged on these five metals only): "
                    f"{parts}.", className="muted")


def _class_scores_text(comparisons: list[ClassComparison]) -> str:
    if not comparisons:
        return "Class forecast so far: no complete lab sample yet."
    twin = score_class_comparisons(comparisons)
    text = f"Class forecast so far ({twin['n']} lab samples): twin Brier {twin['brier']:.2f}, " \
           f"accuracy {twin['accuracy']:.0%}"
    if comparisons[0].reference_probabilities is not None:
        ref = score_class_comparisons(comparisons, reference=True)
        text += f"; last lab value Brier {ref['brier']:.2f}, accuracy {ref['accuracy']:.0%}"
    return text + "."


def _class_station(station: str, summary: views.ClassSummary, comparisons: list[ClassComparison]) -> html.Div:
    cls = summary.most_likely
    heading = html.Div(className="class-heading", children=[
        html.Span(className="class-dot", style={"background": views.CLASS_COLOURS[cls]}),
        html.Strong(station.capitalize()),
        html.Span(f"most likely {cls} ({summary.probabilities[cls]:.0%})" if cls else summary.reason,
                  className="muted"),
    ])
    body = [heading]
    if summary.probabilities:
        body.append(_class_bar(summary.probabilities))
        body.append(html.Div(f"Lutum {summary.lutum_pct:.1f}%, organic matter {summary.om_pct:.1f}% "
                             f"(lab sample {summary.soil_sample_time:%Y-%m-%d})", className="muted"))
        body.append(_reuse_line(summary.probabilities))
    body.append(html.Div(_class_scores_text(comparisons), className="muted"))
    return html.Div(body, className="class-station")


def _reach_facts(values: dict, regime: EntityState | None) -> list[str]:
    facts = []
    if regime is not None:
        facts.append(f"Upstream regime: {regime.values['regime']}")
    if "sediment_load_t_per_d" in values:
        facts.append(f"Discharge into the reach {values['discharge_m3s']:,.0f} m³/s "
                     f"({values['share']:.0%} of the upstream discharge)")
        facts.append(f"Suspended matter {values['spm_mg_l']:.0f} mg/l (from {values['spm_source']}), "
                     f"sediment load {values['sediment_load_t_per_d']:,.0f} t/day")
    else:
        facts.append(f"No flux at this step (suspended matter: {values.get('spm_source', 'none')})")
    return facts


def _reach_table(values: dict) -> html.Table:
    flux = values.get("metal_flux_kg_per_d", {})
    rows = [html.Tr([html.Th("metal"), html.Th("flux (kg/day)"), html.Th("deposit median (mg/kg)"),
                     html.Th("deposit 90% range")])]
    for m, median in values["deposit_median"].items():
        rows.append(html.Tr([
            html.Td(m.upper()),
            html.Td(f"{flux[m]:,.1f}" if m in flux else "–"),
            html.Td(f"{median:.3g}"),
            html.Td(f"{values['deposit_p05'][m]:.3g} – {values['deposit_p95'][m]:.3g}"),
        ]))
    return html.Table(rows, className="reach-table")


def _reach_panel(label: str, reach: EntityState | None, regime: EntityState | None) -> html.Div:
    """Flux and fresh deposit of one reach, from its entity state and its upstream station's regime."""
    if reach is None:
        return html.Div([html.Strong(label), html.Div("No data for this reach yet.", className="muted")],
                        className="reach-panel")
    v = reach.values
    cls = v.get("class")
    heading = html.Div(className="class-heading", children=[
        html.Span(className="class-dot", style={"background": views.CLASS_COLOURS.get(cls)}),
        html.Strong(label),
        html.Span(f"{reach.time_utc:%Y-%m-%d %H:%M} UTC", className="muted"),
    ])
    body = [heading, html.Ul([html.Li(f) for f in _reach_facts(v, regime)], className="reach-facts")]
    if "deposit_median" not in v:
        body.append(html.Div("Fresh deposit: no sediment has arrived yet.", className="muted"))
        return html.Div(body, className="reach-panel")
    body.append(html.Div(f"Fresh deposit: {cls}" if cls else f"Fresh deposit: {v.get('class_reason', 'no class')}",
                         className="reach-subhead"))
    if "class_probabilities" in v:
        body.append(_class_bar(v["class_probabilities"]))
        body.append(html.Div(f"Load-weighted lutum {v['deposit_lutum_pct']:.1f}%, organic carbon "
                             f"{v['deposit_corg_pct']:.1f}%", className="muted"))
        body.append(_reuse_line(v["class_probabilities"]))
    body.append(_reach_table(v))
    return html.Div(body, className="reach-panel")


def _class_legend() -> html.Div:
    return html.Div(className="class-legend muted", children=[
        html.Span([html.Span(className="class-dot", style={"background": views.CLASS_COLOURS[c]}), c])
        for c in QUALITY_CLASSES
    ])


def create_app(
    store: StateStore,
    network: RiverNetwork,
    metals: tuple[str, ...],
    limits: pd.DataFrame | None = None,
    refresh_ms: int = 1000,
    units: dict[str, str] | None = None,
    features: tuple[str, ...] = (),
    soil_correction: pd.DataFrame | None = None,
    om_per_corg: float | None = None,
) -> Dash:
    """`units`: sensor_id -> unit (panel titles; a station without a sensor gets its selector disabled).
    `features`: the soft sensor's inputs, marked as "model input" in the sensor selectors.
    `soil_correction` and `om_per_corg` (with `limits`) are needed for the quality-class panel."""
    if limits is None or soil_correction is None or om_per_corg is None:
        raise ValueError("The quality-class panel needs limits, soil_correction and om_per_corg.")
    modelled = [s.id for s in network.modelled_stations()]
    series = views.EstimateSeries()
    daily = views.DailyStats()
    units = units or {}
    app = Dash(__name__, title=TITLE, assets_folder=str(Path(__file__).parent / "assets"))

    def default_option(kind: str) -> str:
        """The longest-window option that is a model input, else the first option."""
        title, prefix, options = SENSOR_PANELS[kind]
        used = [o for o in options if _feature_name(prefix, o) in features]
        return used[-1] if used else next(iter(options))

    sensor_controls = [
        _control(title, dcc.Dropdown(id=f"view-{kind}", value=default_option(kind), clearable=False,
                                     style={"width": "230px"}))
        for kind, (title, _, _) in SENSOR_PANELS.items()
    ]

    app.layout = html.Div(className="page", children=[
        html.Header(className="header", children=[
            html.H1(TITLE),
            html.Div(id="sim-clock", className="clock"),
        ]),
        html.Section(className="card", children=[
            html.H3("River network: suspended sediment at the stations, fresh deposit along the reaches"),
            _class_legend(),
            html.Div(className="grid-2", children=[
                dcc.Graph(id="network-map", style={"height": "480px"},
                          config={"displaylogo": False, "scrollZoom": True}),
                html.Div(id="reach-details"),
            ]),
        ]),
        html.Section(className="card", children=[
            html.Div(className="controls", children=[
                _control("Station", dcc.Dropdown(
                    id="station", options=[{"label": s.capitalize(), "value": s} for s in modelled],
                    value=modelled[0] if modelled else None, clearable=False, style={"width": "170px"})),
                _control("Metal", dcc.Dropdown(
                    id="metal", options=[{"label": m.upper(), "value": m} for m in metals],
                    value=metals[0], clearable=False, style={"width": "110px"})),
                html.Div(className="divider"),
                *sensor_controls,
            ]),
            dcc.Graph(id="metal-ts", style={"height": "900px"}, config={"displaylogo": False}),
        ]),
        html.Section(className="card", children=[
            html.H3("Accuracy so far: twin vs. last known lab value"),
            html.Div(className="grid-2", children=[
                dash_table.DataTable(
                    id="scoreboard", style_cell=_TABLE_CELL, style_header=_TABLE_HEADER,
                    style_table={"overflowX": "auto"},
                    style_data_conditional=[
                        {"if": {"filter_query": '{verdict} = "twin better"'}, "backgroundColor": "#e6f4ea"},
                        {"if": {"filter_query": '{verdict} = "last lab value better"'},
                         "backgroundColor": "#fdecea"}]),
                dcc.Graph(id="running-error", style={"height": "300px"}, config={"displaylogo": False}),
            ]),
        ]),
        html.Section(className="card", children=[
            html.H3("Quality class of the suspended sediment (Rbk 2022)"),
            _class_legend(),
            html.Div(id="class-probs", className="class-grid"),
        ]),
        html.Section(className="card", children=[
            html.H3("Alerts"),
            dash_table.DataTable(
                id="alerts",
                columns=[{"name": c, "id": c} for c in ("time", "station", "metal", "kind", "severity", "message")],
                page_size=10, style_cell=_TABLE_CELL, style_header=_TABLE_HEADER,
                style_table={"overflowX": "auto"},
            ),
        ]),
        dcc.Interval(id="tick", interval=refresh_ms),
        dcc.Store(id="live-tick"),
    ])

    # Redrawing a chart while the user drags a zoom box cancels the drag, and with a refresh every second
    # that happened almost every time. The tick therefore passes through this browser-side gate, which
    # holds it back while a mouse button / finger is down on a chart (flag set in assets/interaction.js).
    # After release the zoom is applied, and `uirevision` keeps it across later refreshes.
    app.clientside_callback(
        """function (n) {
            const t = window.rdtPointerDownSince;
            if (t && Date.now() - t < 15000) { return window.dash_clientside.no_update; }
            return n;
        }""",
        Output("live-tick", "data"),
        Input("tick", "n_intervals"),
    )

    @app.callback(
        *[Output(f"view-{kind}", "options") for kind in SENSOR_PANELS],
        *[Output(f"view-{kind}", "disabled") for kind in SENSOR_PANELS],
        Input("station", "value"),
    )
    def sensor_options(station):
        """Options for the selected station: model inputs marked, selector off if the station lacks the sensor."""
        opts, disabled = [], []
        for kind, (_, prefix, options) in SENSOR_PANELS.items():
            opts.append([{"label": f"{label}  · model input" if _feature_name(prefix, o) in features else label,
                          "value": o} for o, label in options.items()])
            disabled.append(bool(units) and f"{station}_{kind}" not in units)
        return *opts, *disabled

    @app.callback(
        Output("sim-clock", "children"),
        Output("metal-ts", "figure"),
        Output("alerts", "data"),
        Output("scoreboard", "data"),
        Output("scoreboard", "columns"),
        Output("running-error", "figure"),
        Input("live-tick", "data"),
        Input("station", "value"),
        Input("metal", "value"),
        *[Input(f"view-{kind}", "value") for kind in SENSOR_PANELS],
    )
    def refresh(_n, station, metal, *sensor_views):
        choices = dict(zip(SENSOR_PANELS, sensor_views))
        now = store.sim_time()
        clock = f"Simulated time  {now:%Y-%m-%d %H:%M} UTC" if now else "Waiting for data..."
        metal_fig = views.metal_timeseries(
            metal, series.get(f"twin/{station}", store.estimates(station), metal),
            store.readings(f"{station}_spm_{metal}"), limits,
            reference=series.get(f"ref/{station}", store.reference_estimates(station), metal),
        )
        comparisons = store.comparisons(station)
        board = views.scoreboard_rows(comparisons, metals)
        cols = [{"name": c, "id": c} for c in dict.fromkeys(k for row in board for k in row)]
        panels = []
        for kind, (title, _, options) in SENSOR_PANELS.items():
            sid = f"{station}_{kind}"
            if units and sid not in units:
                continue
            option = choices[kind] if choices[kind] in options else default_option(kind)
            stat, days = option.split("_")
            bins = daily.get(sid, lambda since, sid=sid: store.readings(sid, since=since),
                             keep=lambda v, kind=kind: passes_arrival_rules(kind, v))
            x, y = views.window_series(bins, stat, int(days))
            unit = units.get(sid)
            panels.append((kind, f"{title}, {options[option]}" + (f" ({unit})" if unit else ""), x, y))
        fig = views.with_sensor_panels(metal_fig, panels)
        fig["layout"]["uirevision"] = station          # keep the zoom when switching metal or sensor view
        return (clock, fig, views.alerts_rows(store.alerts()), board, cols,
                views.running_error(comparisons, metal))

    # Class probabilities take ~10 ms per estimate; recompute only when the estimate or soil values change.
    summaries: dict[str, tuple[tuple, views.ClassSummary]] = {}

    def station_summary(station: str) -> views.ClassSummary:
        estimates = store.estimates(station)
        est = estimates[-1] if estimates else None
        lutum, corg = store.latest_reading(f"{station}_spm_lutum"), store.latest_reading(f"{station}_spm_corg")
        key = (est.time_utc if est else None, lutum, corg)
        cached = summaries.get(station)
        if cached is None or cached[0] != key:
            summaries[station] = (key, views.class_summary(est, lutum, corg, limits, soil_correction, om_per_corg))
        return summaries[station][1]

    @app.callback(Output("class-probs", "children"), Input("live-tick", "data"))
    def class_panel(_n):
        return [_class_station(s, station_summary(s), store.class_comparisons(s)) for s in modelled]

    @app.callback(
        Output("network-map", "figure"),
        Output("reach-details", "children"),
        Input("live-tick", "data"),
        Input("network-map", "clickData"),
    )
    def network_panel(_n, click):
        if not network.reaches:
            return views.network_map(network, {}, {}, {}), html.Div("No reaches configured.", className="muted")
        station_class = {s: station_summary(s).most_likely for s in modelled}
        reaches, regimes = store.entities("reach"), store.entities("regime")
        reach_class = {rid: e.values.get("class") for rid, e in reaches.items()}
        fig = views.network_map(network, station_class, reach_class,
                                views.map_hover(network, station_class, regimes, reaches))
        selected = _clicked_reach(click) or network.reaches[0].id
        reach = next(r for r in network.reaches if r.id == selected)
        details = _reach_panel(views.reach_label(network, reach), reaches.get(reach.id),
                               regimes.get(reach.upstream))
        return fig, details

    return app


def _clicked_reach(click: dict | None) -> str | None:
    """Reach id of the clicked map point (see `views.network_map`), or None."""
    points = (click or {}).get("points") or []
    return points[0].get("customdata") if points else None
