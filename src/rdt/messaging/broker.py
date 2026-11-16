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


class MqttBroker:
    """MQTT-backed broker (paho-mqtt, e.g. against a local Mosquitto).

    TODO: implement with paho.mqtt.client.Client:
      - publish: client.publish(topic, message.to_json(), qos=1)
      - subscribe: client.subscribe(pattern); in on_message decode with SensorMessage.from_json and
        dispatch to handlers whose pattern matches (`topic_matches`)
      - run client.loop_start() in __init__, loop_stop()/disconnect() in close()
    """

    def __init__(self, host: str = "localhost", port: int = 1883) -> None:
        raise NotImplementedError("MQTT broker not implemented yet; use broker.kind: memory")

    def publish(self, topic: str, message: SensorMessage) -> None:
        raise NotImplementedError

    def subscribe(self, pattern: str, handler: Handler) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


def make_broker(cfg: BrokerConfig) -> Broker:
    if cfg.kind == "memory":
        return InMemoryBroker()
    if cfg.kind == "mqtt":
        return MqttBroker(**cfg.mqtt)
    raise ValueError(f"Unknown broker kind: {cfg.kind}")
