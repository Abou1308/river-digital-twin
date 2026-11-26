"""MQTT broker (rdt.messaging.broker.MqttBroker): logic against a fake paho client, and a round trip through
a real broker when one runs on localhost:1883 (e.g. `docker compose up mosquitto`)."""

import socket
import threading
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from rdt.messaging import MqttBroker, SensorMessage
from rdt.messaging.broker import STATUS_PREFIX

MSG = SensorMessage("lobith_spm_cd", datetime(2021, 1, 1, tzinfo=timezone.utc), 1.5,
                    available_utc=datetime(2021, 1, 15, tzinfo=timezone.utc))


class FakeInfo:
    rc = 0

    def __init__(self):
        self.waited = 0

    def wait_for_publish(self, timeout=None):
        self.waited += 1


class FakeClient:
    """Records calls; `loop_start` answers the connection like a broker would."""

    on_subscribe = None

    def __init__(self, accept=True, confirm=True):
        self.accept, self.confirm = accept, confirm
        self.published, self.subscribed, self.will = [], [], None

    def will_set(self, topic, payload, qos, retain):
        self.will = (topic, payload, retain)

    def connect(self, host, port):
        self.address = (host, port)

    def loop_start(self):
        self.on_connect(self, None, {}, SimpleNamespace(is_failure=not self.accept), None)

    def loop_stop(self):
        pass

    def disconnect(self):
        pass

    def subscribe(self, pattern, qos):
        self.subscribed.append(pattern)
        mid = len(self.subscribed)
        if self.on_subscribe is not None and self.confirm:
            self.on_subscribe(self, None, mid, [], None)
        return 0, mid

    def publish(self, topic, payload, qos, retain=False):
        self.published.append((topic, payload, retain))
        return FakeInfo()

    def deliver(self, topic, payload):
        self.on_message(self, None, SimpleNamespace(topic=topic, payload=payload.encode()))


def test_publish_and_dispatch_with_fake_client():
    client = FakeClient()
    broker = MqttBroker(client=client, will=("twin", "offline"))
    assert client.will == (STATUS_PREFIX + "twin", "offline", True)
    assert client.subscribed == [STATUS_PREFIX + "#"]
    got = []
    broker.subscribe("rws/+/spm/#", lambda topic, m: got.append((topic, m)))
    broker.subscribe("rws/+/spm/#", lambda topic, m: got.append(("again", m)))
    assert client.subscribed.count("rws/+/spm/#") == 1                 # one MQTT subscription per pattern
    broker.publish("rws/lobith/spm/cd", MSG)
    topic, payload, retain = client.published[-1]
    assert topic == "rws/lobith/spm/cd" and SensorMessage.from_json(payload) == MSG and not retain
    client.deliver("rws/lobith/spm/cd", MSG.to_json())
    client.deliver("rws/lobith/turbidity", MSG.to_json())
    assert got == [("rws/lobith/spm/cd", MSG), ("again", MSG)]


def test_status_and_resubscribe_on_reconnect():
    client = FakeClient()
    broker = MqttBroker(client=client)
    broker.subscribe("rws/#", lambda topic, m: None)
    assert not broker.wait_for_status("twin", "ready", timeout_s=0.01)
    client.deliver(STATUS_PREFIX + "twin", "ready")
    assert broker.wait_for_status("twin", "ready", timeout_s=0.01)
    broker.publish_status("replay", "finished")
    assert client.published[-1] == (STATUS_PREFIX + "replay", "finished", True)
    client.subscribed.clear()
    client.loop_start()                                                  # reconnect
    assert client.subscribed == [STATUS_PREFIX + "#", "rws/#"]


def test_unconfirmed_subscription_is_an_error():
    broker = MqttBroker(client=FakeClient(confirm=False), connect_timeout_s=0.01)
    with pytest.raises(TimeoutError, match="did not confirm"):
        broker.subscribe("rws/#", lambda topic, m: None)


def test_refused_connection_is_an_error():
    with pytest.raises(ConnectionError, match="did not accept"):
        MqttBroker(client=FakeClient(accept=False))


def _broker_running(host="localhost", port=1883) -> bool:
    try:
        socket.create_connection((host, port), timeout=0.5).close()
        return True
    except OSError:
        return False


@pytest.mark.skipif(not _broker_running(), reason="no MQTT broker on localhost:1883")
def test_round_trip_through_a_real_broker():
    """Messages arrive complete and in order; a retained status reaches a client that subscribes later."""
    prefix = f"rws/test-{uuid.uuid4().hex[:8]}"
    receiver, sender = MqttBroker(), MqttBroker()
    got, done = [], threading.Event()

    def handle(topic, m):
        got.append(m)
        if len(got) == 1200:
            done.set()

    receiver.subscribe(prefix + "/#", handle)
    sent = [SensorMessage(f"s{i % 3}", MSG.time_utc, float(i)) for i in range(1200)]
    for i, m in enumerate(sent):
        sender.publish(f"{prefix}/{i % 3}", m)
    assert done.wait(10)
    assert got == sent
    name = f"test-{uuid.uuid4().hex[:8]}"
    sender.publish_status(name, "ready")
    late = MqttBroker()
    assert late.wait_for_status(name, "ready", timeout_s=5)
    sender.publish_status(name, "")                                      # clear the retained status
    for b in (receiver, sender, late):
        b.close()


def test_a_failing_handler_stops_and_is_reported():
    client = FakeClient()
    broker = MqttBroker(client=client)

    def handler(topic, m):
        raise ValueError("bad reading")

    broker.subscribe("rws/#", handler)
    assert broker.error is None
    with pytest.raises(ValueError, match="bad reading"):                # propagates into paho's thread
        client.deliver("rws/lobith/turbidity", MSG.to_json())
    assert str(broker.error) == "bad reading"
    broker.wait_for_failure()                                            # returns at once
    with pytest.raises(RuntimeError, match="handling stopped: bad reading"):
        broker.wait_for_status("replay", "finished")
    with pytest.raises(RuntimeError, match="handling stopped"):
        broker.raise_if_failed()
