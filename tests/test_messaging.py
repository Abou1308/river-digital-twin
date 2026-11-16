from datetime import datetime, timezone

import pytest

from rdt.messaging import InMemoryBroker, SensorMessage, topic_matches


@pytest.mark.parametrize(
    ("pattern", "topic", "expected"),
    [
        ("rws/#", "rws/lobith/spm/cd", True),
        ("rws/+/turbidity", "rws/lobith/turbidity", True),
        ("rws/+/turbidity", "rws/lobith/spm/cd", False),
        ("rws/lobith/spm/+", "rws/lobith/spm/cd", True),
        ("rws/lobith", "rws/lobith/spm", False),
        ("#", "anything/at/all", True),
    ],
)
def test_topic_matches(pattern, topic, expected):
    assert topic_matches(pattern, topic) is expected


def test_message_json_roundtrip():
    msg = SensorMessage("lobith_spm_cd", datetime(2021, 1, 1, tzinfo=timezone.utc), 1.23, censored=True, qc="00")
    assert SensorMessage.from_json(msg.to_json()) == msg
    late = SensorMessage("lobith_spm_cd", datetime(2021, 1, 1, tzinfo=timezone.utc), 1.0,
                         available_utc=datetime(2021, 1, 15, tzinfo=timezone.utc))
    assert SensorMessage.from_json(late.to_json()) == late
    assert late.arrival_utc == late.available_utc and msg.arrival_utc == msg.time_utc


def test_in_memory_broker_dispatch():
    broker = InMemoryBroker()
    got = []
    broker.subscribe("rws/+/spm/#", lambda topic, m: got.append(topic))
    msg = SensorMessage("x", datetime(2021, 1, 1, tzinfo=timezone.utc), 1.0)
    broker.publish("rws/lobith/spm/cd", msg)
    broker.publish("rws/lobith/turbidity", msg)
    assert got == ["rws/lobith/spm/cd"]
