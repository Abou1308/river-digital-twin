"""Config loading (rdt.config)."""

from rdt.config import DEFAULT_CONFIG, load_config


def test_config_extends_a_base_file(tmp_path):
    (tmp_path / "base.yaml").write_text(DEFAULT_CONFIG.read_text())
    (tmp_path / "mid.yaml").write_text("extends: base.yaml\nbroker: {kind: mqtt, mqtt: {host: localhost}}\n")
    (tmp_path / "top.yaml").write_text("extends: mid.yaml\nbroker: {mqtt: {host: mosquitto}}\n"
                                       "replay: {tiers: [core]}\n")
    cfg = load_config(tmp_path / "top.yaml")
    assert cfg.broker.kind == "mqtt"
    assert cfg.broker.mqtt == {"host": "mosquitto", "port": 1883}    # nested keys merged
    assert cfg.replay.tiers == ("core",)                               # lists replaced
    assert cfg.replay.speed == load_config().replay.speed              # the rest from the base
