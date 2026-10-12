# -*- coding: utf-8 -*-
"""
Unified message format (architecture decision — Martin).

ALL modules publish into event_queue using this format.
Historical target: Python 3.6.9 (Jetson Nano, JetPack 4.6); compatibility
has not been validated by execution on that runtime.

Message = dict:
{
    "ts":      float   # time.time() when emitted
    "source":  str     # "gesture" | "depth" | "speech" | "system"
    "type":    str     # "event" (detection) | "status" (heartbeat)
    "payload": dict    # module-specific content, see below
}

Payloads expected by the fusion module:
- gesture (SONG): {"gesture": "lay_down"|"squat"|..., "confidence": float}
- depth   (SU)  : {"depth_level": "FAR"|"MIDDLE"|"NEAR"|"NO_TARGET"|"UNAVAILABLE",
                   "near_score": float,          # relative, NOT metric
                   "status": "OK"|"NO_TARGET"|"STALE_PACKET"|"ERROR"}
                  SU's model gives RELATIVE depth, never meters/cm.
- speech  (HUO) : {"keyword": "help", "confidence": float}
- status (all)  : {"state": "alive", "health": str, "backend": str,
                  "reason": optional str, "module": optional str}
                  on health transitions and every 2 s while alive;
                  legacy payloads may omit the added health fields.
                  Alive means process liveness, not backend readiness.
"""
import time

SOURCE_GESTURE = "gesture"
SOURCE_DEPTH = "depth"
SOURCE_SPEECH = "speech"
SOURCE_SYSTEM = "system"

TYPE_EVENT = "event"
TYPE_STATUS = "status"
# Small, frequent payload: person boxes + state, drawn by the UI over
# the live video. Keeps the video fluid — we never ship annotated
# frames between processes.
TYPE_DETECTIONS = "detections"

HEARTBEAT_PERIOD_S = 2.0


def make_message(source, msg_type, payload):
    return {
        "ts": time.time(),
        "source": source,
        "type": msg_type,
        "payload": payload,
    }


def make_event(source, payload):
    return make_message(source, TYPE_EVENT, payload)


def make_detections(source, tracks, fps=0.0):
    return make_message(source, TYPE_DETECTIONS,
                        {"tracks": tracks, "fps": float(fps)})


def make_status(source, state="alive", extra=None):
    payload = {"state": state}
    if extra:
        payload.update(extra)
    return make_message(source, TYPE_STATUS, payload)
