"""Command-line entry point: ``rdt info`` and ``rdt run``."""

from __future__ import annotations

import argparse
import logging
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


def _run(args: argparse.Namespace) -> None:
    from rdt.runtime import build_system

    cfg = load_config(args.config)
    speed = "config" if args.speed is None else (None if args.speed <= 0 else args.speed)
    system = build_system(cfg, speed=speed)

    if args.no_dashboard:
        t0 = time.monotonic()
        n = system.replayer.run()
        print(f"Replayed {n} messages in {time.monotonic() - t0:.1f}s")
        for station in (s.id for s in system.network.modelled_stations()):
            print(f"{station}: {len(system.store.estimates(station))} estimates, "
                  f"{len(system.store.comparisons(station))} lab comparisons")
            from rdt.dashboard.views import scoreboard_rows
            for row in scoreboard_rows(system.store.comparisons(station), cfg.softsensor.metals):
                print("   " + " | ".join(f"{k}: {v}" for k, v in row.items()))
            _print_class_scores(system.store.class_comparisons(station))
        print(f"Alerts: {len(system.store.alerts(limit=10_000))}")
        return

    from rdt.dashboard import create_app

    system.replayer.start()
    app = create_app(system.store, system.network, cfg.softsensor.metals, system.limits, cfg.dashboard.refresh_ms,
                     units={sid: info.unit for sid, info in system.catalog.items()},
                     features=tuple(cfg.softsensor.features), soil_correction=system.soil_correction,
                     om_per_corg=cfg.quality.om_per_corg)
    app.run(host=cfg.dashboard.host, port=cfg.dashboard.port, debug=False)


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

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
