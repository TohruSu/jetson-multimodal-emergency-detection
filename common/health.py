# -*- coding: utf-8 -*-
"""Backend health carried by the existing worker-alive status envelope.

Message timestamps remain wall-clock values. Heartbeat intervals and UI
receipt freshness use separate monotonic clocks and never compare the two.
Backend names and reason codes are fixed descriptions supplied by workers;
exception text, device names, and local paths do not belong in this payload.
"""
import queue
import time

from common.messages import HEARTBEAT_PERIOD_S, make_status


HEALTH_LABELS = {
    "ready": "READY",
    "simulated": "SIMULATION",
    "degraded": "DEGRADED",
    "unavailable": "UNAVAILABLE",
    "disabled": "DISABLED",
    "initializing": "INITIALIZING",
}


class HealthReporter:
    """Publish state changes immediately and repeat the current state every 2s.

    A full event queue drops this status instead of blocking sensor/inference
    work. The next heartbeat retries it; status delivery is best effort.
    Call ``set`` at initialization and whenever backend operation changes.
    ``state=alive`` always describes this process, independently of ``health``.
    """

    def __init__(self, source, event_queue, module=None, clock=None):
        self.source = source
        self.event_queue = event_queue
        self.module = module
        self.clock = clock if clock is not None else time.monotonic
        self._state = None
        self._last_emit = None

    def set(self, health, backend=None, reason=None):
        """Update backend health; return whether the state tuple changed.

        Supplying no reason clears a previous failure reason. Workers should
        provide their fixed backend name on each call (default: ``unknown``).
        """
        if health not in HEALTH_LABELS:
            raise ValueError("Unknown backend health state")
        state = (health, backend or "unknown", reason)
        if state == self._state:
            return False
        self._state = state
        self._emit(self.clock())
        return True

    def heartbeat(self):
        """Repeat health if due; return whether a status was successfully sent."""
        now = self.clock()
        if (self._last_emit is not None
                and now - self._last_emit < HEARTBEAT_PERIOD_S):
            return False
        return self._emit(now)

    def _emit(self, now):
        health, backend, reason = self._state or (
            "initializing", "unknown", None)
        payload = {"health": health, "backend": backend}
        if self.module is not None:
            payload["module"] = self.module
        if reason is not None:
            payload["reason"] = reason
        try:
            self.event_queue.put_nowait(make_status(self.source, extra=payload))
        except queue.Full:
            return False
        self._last_emit = now
        return True


def format_health(payload, fresh=True):
    """Format a UI status without importing Qt or equating alive with ready."""
    label = HEALTH_LABELS.get(payload.get("health"))
    if label is None:
        label = "ALIVE (health unknown)"
    else:
        details = [payload.get("backend") or "unknown"]
        if payload.get("reason"):
            details.append(payload["reason"])
        label += " ({})".format("; ".join(details))
    return label if fresh else "STALE - last " + label
