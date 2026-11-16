"""Dash app. Reads only from the `StateStore` (never from the twin directly), polling on an interval."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
from dash import Dash, Input, Output, dash_table, dcc, html

from rdt.dashboard import views
from rdt.state import StateStore
from rdt.twin.network import RiverNetwork


def create_app(
    store: StateStore,
    network: RiverNetwork,
    metals: tuple[str, ...],
    limits: pd.DataFrame | None = None,
    refresh_ms: int = 1000,
) -> Dash:
    modelled = [s.id for s in network.modelled_stations()]
    series = views.EstimateSeries()
    app = Dash(__name__, title="River Digital Twin")

    app.layout = html.Div(
        style={"fontFamily": "system-ui, sans-serif", "margin": "0 16px"},
        children=[
            html.Div(
                style={"display": "flex", "justifyContent": "space-between", "alignItems": "baseline"},
                children=[html.H2("River Digital Twin"), html.Div(id="sim-clock", style={"fontSize": "1.1em"})],
            ),
            html.Div(
                style={"display": "flex", "gap": "12px", "marginBottom": "8px"},
                children=[
                    dcc.Dropdown(id="station", options=modelled, value=modelled[0] if modelled else None,
                                 clearable=False, style={"width": "200px"}),
                    dcc.Dropdown(id="metal", options=[{"label": m.upper(), "value": m} for m in metals],
                                 value=metals[0], clearable=False, style={"width": "120px"}),
                ],
            ),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "minmax(280px, 1fr) 2fr", "gap": "12px"},
                children=[
                    dcc.Graph(id="map", style={"height": "420px"}),
                    dcc.Graph(id="metal-ts", style={"height": "420px"}),
                ],
            ),
            html.H4("Does the twin beat waiting for the lab? (lab results received so far)"),
            html.Div(style={"fontSize": "0.9em", "color": "#555", "marginBottom": "6px"},
                     children="Each lab result, when it arrives, is compared with what the twin predicted for "
                              "the sample time and with the last lab value that was known then. "
                              "Error = |prediction - lab| / lab. 'About equal' = less than 1 percentage point "
                              "apart. This is a running tally; statistical tests are in "
                              "analysis/04_soft_sensor_evaluation.py."),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "3fr 2fr", "gap": "12px"},
                children=[
                    dash_table.DataTable(id="scoreboard", style_cell={"textAlign": "left", "fontSize": "0.9em"},
                                         style_data_conditional=[
                                             {"if": {"filter_query": '{verdict} = "twin better"'},
                                              "backgroundColor": "#e8f5e9"},
                                             {"if": {"filter_query": '{verdict} = "reference better"'},
                                              "backgroundColor": "#fdecea"}]),
                    dcc.Graph(id="running-error", style={"height": "300px"}),
                ],
            ),
            html.H4("Class probabilities"),
            html.Div(id="class-probs", children="Not available yet: soil-quality class probabilities "
                                                "will be shown here once rdt.quality is implemented."),
            html.H4("Sensors (last 60 days)"),
            dcc.Graph(id="sensor-ts"),
            html.H4("Alerts"),
            dash_table.DataTable(
                id="alerts",
                columns=[{"name": c, "id": c} for c in ("time", "station", "metal", "kind", "severity", "message")],
                page_size=10,
                style_cell={"textAlign": "left", "fontSize": "0.9em"},
            ),
            dcc.Interval(id="tick", interval=refresh_ms),
        ],
    )

    @app.callback(
        Output("sim-clock", "children"),
        Output("map", "figure"),
        Output("metal-ts", "figure"),
        Output("sensor-ts", "figure"),
        Output("alerts", "data"),
        Output("scoreboard", "data"),
        Output("scoreboard", "columns"),
        Output("running-error", "figure"),
        Input("tick", "n_intervals"),
        Input("station", "value"),
        Input("metal", "value"),
    )
    def refresh(_n, station, metal):
        now = store.sim_time()
        clock = f"Simulated time: {now:%Y-%m-%d %H:%M} UTC" if now else "Waiting for data..."
        # TODO: most likely class per station from rdt.quality once implemented
        station_class: dict[str, str | None] = {}
        map_fig = views.network_map(network, station_class)
        metal_fig = views.metal_timeseries(
            metal, series.get(f"twin/{station}", store.estimates(station), metal),
            store.readings(f"{station}_spm_{metal}"), limits,
            reference=series.get(f"ref/{station}", store.reference_estimates(station), metal),
        )
        comparisons = store.comparisons(station)
        board = views.scoreboard_rows(comparisons, metals)
        cols = [{"name": c, "id": c} for c in dict.fromkeys(k for row in board for k in row)]
        since = now - timedelta(days=60) if now else None
        sensors = {name: store.readings(f"{station}_{name}", since=since)
                   for name in ("discharge", "turbidity", "spm_daily")}
        sensor_fig = views.sensor_timeseries(sensors, now)
        return (clock, map_fig, metal_fig, sensor_fig, views.alerts_rows(store.alerts()), board, cols,
                views.running_error(comparisons, metal))

    return app
