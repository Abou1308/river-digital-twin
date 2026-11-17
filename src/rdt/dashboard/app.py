"""Dash app. Reads only from the `StateStore` (never from the twin directly), polling on an interval.

What the dashboard shows (kept here rather than as text on the page):

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
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from dash import Dash, Input, Output, dash_table, dcc, html

from rdt.dashboard import views
from rdt.data.cleaning import passes_arrival_rules
from rdt.state import StateStore
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


def create_app(
    store: StateStore,
    network: RiverNetwork,
    metals: tuple[str, ...],
    limits: pd.DataFrame | None = None,
    refresh_ms: int = 1000,
    units: dict[str, str] | None = None,
    features: tuple[str, ...] = (),
) -> Dash:
    """`units`: sensor_id -> unit (panel titles; a station without a sensor gets its selector disabled).
    `features`: the soft sensor's inputs, marked as "model input" in the sensor selectors."""
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
            html.H3("Class probabilities"),
            html.Div(id="class-probs", className="muted",
                     children="Not available yet: soil-quality class probabilities will be shown here "
                              "once rdt.quality is implemented."),
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

    return app
