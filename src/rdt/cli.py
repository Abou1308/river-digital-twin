"""Command-line entry point: ``rdt info``, ``rdt run`` (replay and twin in one process), and ``rdt twin``
with ``rdt replay`` (two processes connected over MQTT)."""

from __future__ import annotations

import _thread
import argparse
import logging
import threading
import time

from rdt.config import load_config


def _info(args: argparse.Namespace) -> None:
    from rdt.data import load_model_table, load_sensors

    cfg = load_config(args.config)
    sensors = load_sensors(cfg.data_dir)
    table = load_model_table(cfg.data_dir)
    print(f"Data dir: {cfg.data_dir}")
    print(sensors.groupby(["station", "type", "tier"]).size().rename("sensors").to_string())
    print()
    print(table.groupby(["station", "period"])["sample_time_utc"].agg(["count", "min", "max"]).to_string())


def _print_class_scores(comparisons: list) -> None:
    from rdt.monitoring.classes import score_class_comparisons

    if not comparisons:
        print("   classes: no complete lab samples")
        return
    for name, ref in (("twin", False), ("reference", True)):
        if ref and comparisons[0].reference_probabilities is None:
            continue
        s = score_class_comparisons(comparisons, reference=ref)
        print(f"   classes ({name}): n={s['n']} Brier={s['brier']:.3f} accuracy={s['accuracy']:.0%} "
              f"confusion (rows observed, columns predicted)={s['confusion'].tolist()}")


def _speed(args: argparse.Namespace) -> float | None | str:
    return "config" if args.speed is None else (None if args.speed <= 0 else args.speed)


def _print_summary(side) -> None:
    from rdt.dashboard.views import scoreboard_rows

    for station in (s.id for s in side.network.modelled_stations()):
        print(f"{station}: {len(side.store.estimates(station))} estimates, "
              f"{len(side.store.comparisons(station))} lab comparisons")
        for row in scoreboard_rows(side.store.comparisons(station), side.cfg.softsensor.metals):
            print("   " + " | ".join(f"{k}: {v}" for k, v in row.items()))
        _print_class_scores(side.store.class_comparisons(station))
    print(f"Alerts: {len(side.store.alerts(limit=10_000))}")


def _serve_dashboard(side) -> None:
    from rdt.dashboard import create_app

    cfg = side.cfg
    app = create_app(side.store, side.network, cfg.softsensor.metals, side.limits, cfg.dashboard.refresh_ms,
                     units={sid: info.unit for sid, info in side.catalog.items()},
                     features=tuple(cfg.softsensor.features), soil_correction=side.soil_correction,
                     om_per_corg=cfg.quality.om_per_corg)
    app.run(host=cfg.dashboard.host, port=cfg.dashboard.port, debug=False)


def _mqtt_broker(cfg, command: str, **kwargs):
    from rdt.messaging import MqttBroker

    if cfg.broker.kind != "mqtt":
        raise SystemExit(f"rdt {command} talks to the other process over MQTT: set broker.kind: mqtt in the "
                         f"config (e.g. --config config/docker.yaml), and start a broker (docker compose up mosquitto).")
    return MqttBroker(**cfg.broker.mqtt, **kwargs)


def _run(args: argparse.Namespace) -> None:
    from rdt.runtime import build_system

    system = build_system(load_config(args.config), speed=_speed(args))
    if args.no_dashboard:
        t0 = time.monotonic()
        n = system.replayer.run()
        print(f"Replayed {n} messages in {time.monotonic() - t0:.1f}s")
        _print_summary(system)
        return
    system.replayer.start()
    _serve_dashboard(system)


def _twin(args: argparse.Namespace) -> None:
    """The twin and dashboard, fed by a separate `rdt replay` over MQTT."""
    from rdt.runtime import build_twin

    cfg = load_config(args.config)
    side = build_twin(cfg)
    broker = _mqtt_broker(cfg, "twin", will=("twin", "offline"))
    side.twin.attach(broker)
    broker.publish_status("twin", "ready")
    logging.getLogger(__name__).info("Twin listening on MQTT; waiting for rdt replay")
    try:
        if args.no_dashboard:
            broker.wait_for_status("replay", "finished")
            _print_summary(side)
        else:
            threading.Thread(target=_stop_on_failure, args=(broker,), daemon=True).start()
            _serve_dashboard(side)
            broker.raise_if_failed()
    finally:
        if not broker.failed:                 # after a failure the broker's last will reports "offline"
            broker.publish_status("twin", "offline")
        broker.close()


def _stop_on_failure(broker) -> None:
    """Stop the dashboard server (in the main thread) once the twin can no longer handle messages, so the
    process ends with the error instead of serving a state that has stopped updating."""
    broker.wait_for_failure()
    logging.getLogger(__name__).error("The twin stopped handling messages; shutting down the dashboard")
    _thread.interrupt_main()


def _replay(args: argparse.Namespace) -> None:
    """Publish the replay period to MQTT once a twin (`rdt twin`) is listening."""
    from rdt.data import load_sensors, sensor_catalog
    from rdt.runtime import build_replayer

    cfg = load_config(args.config)
    broker = _mqtt_broker(cfg, "replay")
    log = logging.getLogger(__name__)
    log.info("Waiting for the twin to announce it is listening (rdt twin)")
    broker.wait_for_status("twin", "ready")
    replayer = build_replayer(cfg, sensor_catalog(load_sensors(cfg.data_dir)), broker, _speed(args))
    t0 = time.monotonic()
    n = replayer.run()
    broker.publish_status("replay", "finished", retain=False)
    broker.close()
    print(f"Published {n} messages in {time.monotonic() - t0:.1f}s")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rdt", description="River Digital Twin")
    parser.add_argument("--config", default=None, help="YAML config (default: config/default.yaml)")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="Summarise the dataset").set_defaults(func=_info)

    run = sub.add_parser("run", help="Replay data through the twin (with dashboard by default)")
    run.add_argument("--speed", type=float, default=None,
                     help="Simulated seconds per wall second (0 = as fast as possible; default from config)")
    run.add_argument("--no-dashboard", action="store_true", help="Run headless and print a summary")
    run.set_defaults(func=_run)

    twin = sub.add_parser("twin", help="Run the twin (with dashboard by default), fed over MQTT by rdt replay")
    twin.add_argument("--no-dashboard", action="store_true",
                      help="Run headless and print a summary when the replay has finished")
    twin.set_defaults(func=_twin)

    replay = sub.add_parser("replay", help="Publish the replay period to MQTT once rdt twin is listening")
    replay.add_argument("--speed", type=float, default=None,
                        help="Simulated seconds per wall second (0 = as fast as possible; default from config)")
    replay.set_defaults(func=_replay)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
