"""Publish/subscribe broker abstraction.

Topics follow MQTT conventions (``rws/<station>/<quantity>``, taken from sensors.csv) so the
in-memory broker used for development and tests can be swapped for a real MQTT broker without
touching publishers or subscribers.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Protocol

from rdt.config import BrokerConfig
from rdt.messaging.messages import SensorMessage

Handler = Callable[[str, SensorMessage], None]


def topic_matches(pattern: str, topic: str) -> bool:
    """MQTT-style matching: ``+`` matches one level, ``#`` (last) matches any remaining levels."""
    p_parts, t_parts = pattern.split("/"), topic.split("/")
    for i, p in enumerate(p_parts):
        if p == "#":
            return i == len(p_parts) - 1
        if i >= len(t_parts) or (p != "+" and p != t_parts[i]):
            return False
    return len(p_parts) == len(t_parts)


class Broker(Protocol):
    def publish(self, topic: str, message: SensorMessage) -> None: ...
    def subscribe(self, pattern: str, handler: Handler) -> None: ...
    def close(self) -> None: ...


class InMemoryBroker:
    """Synchronous, thread-safe broker: handlers run in the publisher's thread, in subscription order."""

    def __init__(self) -> None:
        self._subs: list[tuple[str, Handler]] = []
        self._lock = threading.Lock()

    def publish(self, topic: str, message: SensorMessage) -> None:
        with self._lock:
            subs = list(self._subs)
        for pattern, handler in subs:
            if topic_matches(pattern, topic):
                handler(topic, message)

    def subscribe(self, pattern: str, handler: Handler) -> None:
        with self._lock:
            self._subs.append((pattern, handler))

    def close(self) -> None:
        with self._lock:
            self._subs.clear()


STATUS_PREFIX = "rdt/status/"
_PUBLISH_WINDOW = 500      # publishes between waits for the broker's acknowledgement


class MqttBroker:
    """MQTT-backed broker (paho-mqtt), e.g. against a local Mosquitto (docker/mosquitto.conf).

    Sensor messages travel as JSON (`SensorMessage.to_json`) at QoS 1, so the broker acknowledges each
    one. Handlers run in paho's network thread, one message at a time in arrival order; MQTT keeps the
    order of one publisher's messages, which the twin relies on (readings arrive in time order).
    `publish` waits for the broker's acknowledgement every `_PUBLISH_WINDOW` messages, so a fast replay
    cannot queue the whole dataset in memory.

    Besides sensor messages, processes exchange short status strings on ``rdt/status/<name>``
    (`publish_status`, `wait_for_status`): the twin announces that it is listening before the replay
    starts publishing, and the replay announces that it has finished. `will` is a status (name, value)
    the broker publishes if this client disconnects without closing, e.g. ("twin", "offline").
    """

    def __init__(self, host: str = "localhost", port: int = 1883, qos: int = 1, connect_timeout_s: float = 10.0,
                 will: tuple[str, str] | None = None, client=None) -> None:
        if client is None:
            import paho.mqtt.client as mqtt     # optional dependency: pip install -e ".[mqtt]"

            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.qos = qos
        self.timeout = connect_timeout_s
        self._client = client
        self._subs: list[tuple[str, Handler]] = []
        self._lock = threading.Lock()
        self._status: dict[str, str] = {}
        self._status_changed = threading.Condition()
        self._connected = threading.Event()
        self._acked: set[int] = set()                 # message ids of acknowledged subscriptions
        self._suback = threading.Condition()
        self._connect_error: str | None = None
        self._sent = 0
        self._last = None
        client.on_connect = self._on_connect
        client.on_message = self._on_message
        client.on_subscribe = self._on_subscribe
        if will is not None:
            client.will_set(STATUS_PREFIX + will[0], will[1], qos=qos, retain=True)
        try:
            client.connect(host, port)
        except OSError as e:
            raise ConnectionError(f"Cannot reach the MQTT broker at {host}:{port}: {e}") from e
        client.loop_start()
        if not self._connected.wait(connect_timeout_s) or self._connect_error:
            client.loop_stop()
            raise ConnectionError(f"MQTT broker at {host}:{port} did not accept the connection "
                                  f"({self._connect_error or f'no answer within {connect_timeout_s:.0f} s'}).")

    def _on_connect(self, client, userdata, flags, reason_code, properties) -> None:
        if reason_code.is_failure:
            self._connect_error = str(reason_code)
        else:
            # Subscriptions do not survive a reconnect with a clean session, so (re)subscribe here.
            with self._lock:
                patterns = dict.fromkeys(p for p, _ in self._subs)
            for pattern in (STATUS_PREFIX + "#", *patterns):
                client.subscribe(pattern, qos=self.qos)
        self._connected.set()

    def _on_subscribe(self, client, userdata, mid, reason_codes, properties) -> None:
        with self._suback:
            self._acked.add(mid)
            self._suback.notify_all()

    def _on_message(self, client, userdata, msg) -> None:
        if msg.topic.startswith(STATUS_PREFIX):
            with self._status_changed:
                self._status[msg.topic.removeprefix(STATUS_PREFIX)] = msg.payload.decode()
                self._status_changed.notify_all()
            return
        message = SensorMessage.from_json(msg.payload)
        with self._lock:
            subs = list(self._subs)
        for pattern, handler in subs:
            if topic_matches(pattern, msg.topic):
                handler(msg.topic, message)

    def publish(self, topic: str, message: SensorMessage) -> None:
        self._last = self._send(topic, message.to_json())
        self._sent += 1
        if self._sent % _PUBLISH_WINDOW == 0:
            self._last.wait_for_publish(self.timeout)

    def subscribe(self, pattern: str, handler: Handler) -> None:
        """Returns once the broker has confirmed the subscription, so messages published from then on
        (by any client) reach `handler`."""
        with self._lock:
            new = all(p != pattern for p, _ in self._subs)
            self._subs.append((pattern, handler))
        if not new:
            return
        rc, mid = self._client.subscribe(pattern, qos=self.qos)
        if rc != 0:
            raise RuntimeError(f"MQTT subscribe to {pattern} failed (paho error code {rc}).")
        with self._suback:
            if not self._suback.wait_for(lambda: mid in self._acked, self.timeout):
                raise TimeoutError(f"MQTT broker did not confirm the subscription to {pattern} "
                                   f"within {self.timeout:.0f} s.")

    def publish_status(self, name: str, value: str, retain: bool = True) -> None:
        """Publish a status; retained by default, so a process that subscribes later still sees it.
        Waits until earlier messages from this client have been acknowledged too."""
        if self._last is not None:
            self._last.wait_for_publish(self.timeout)
        self._send(STATUS_PREFIX + name, value, retain=retain).wait_for_publish(self.timeout)

    def wait_for_status(self, name: str, value: str, timeout_s: float | None = None) -> bool:
        """Block until status `name` equals `value`; False if `timeout_s` passes first."""
        with self._status_changed:
            return self._status_changed.wait_for(lambda: self._status.get(name) == value, timeout_s)

    def close(self) -> None:
        if self._last is not None:
            self._last.wait_for_publish(self.timeout)
        self._client.disconnect()
        self._client.loop_stop()

    def _send(self, topic: str, payload: str, retain: bool = False):
        info = self._client.publish(topic, payload, qos=self.qos, retain=retain)
        if info.rc != 0:
            raise RuntimeError(f"MQTT publish to {topic} failed (paho error code {info.rc}).")
        return info


def make_broker(cfg: BrokerConfig) -> Broker:
    if cfg.kind == "memory":
        return InMemoryBroker()
    if cfg.kind == "mqtt":
        return MqttBroker(**cfg.mqtt)
    raise ValueError(f"Unknown broker kind: {cfg.kind}")
