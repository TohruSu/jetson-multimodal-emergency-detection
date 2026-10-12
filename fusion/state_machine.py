# -*- coding: utf-8 -*-
"""
MODULE 4 — State fusion (owner: Martin)

State machine combining the three modules' events and generating the
alert sentence (the "SMS" shown on screen).

States:
    IDLE ----(gesture lay_down)----> ALERT_L1
    ALERT_L1 --(valid help, 0 <= delay <= 10 s)--> ALERT_L2
    ALERT_L1/L2 --(> 30 s without relevant activity)--> IDLE

Confirmed fall events start an incident. Duplicate falls preserve its first
timestamp and severity; valid help or another fall refreshes inactivity.
Help requires a recognized keyword and a finite confidence in [0.50, 1.0].
Valid help after the escalation window may refresh an active alert, but
cannot raise its level. Unrelated, invalid, and pre-fall speech is ignored.
Fall/help timestamps must be finite numbers; missing timestamps use receipt
time. Older falls cannot replace a newer incident's fall location, and
out-of-order duplicates retain the most recent timestamped fall location.
Reset rejects fall/help emitted at or before the reset cutoff. A manual
reset uses its time; expiration uses the incident's inactivity deadline,
so a newer fall can start an incident even when its queue delivery is late.

The current position (depth module) is memorized continuously and
injected into the alert sentence.

Runs in the MAIN PROCESS (called by the UI QTimer), so there are no
synchronization issues.
"""
import math
from numbers import Real
import time

from common.messages import (SOURCE_GESTURE, SOURCE_DEPTH, SOURCE_SPEECH,
                             TYPE_EVENT)
from modules.speech_huo import KEYWORDS

STATE_IDLE = "IDLE"
STATE_ALERT_L1 = "ALERT_L1"
STATE_ALERT_L2 = "ALERT_L2"

KEYWORD_WINDOW_S = 10.0   # max delay gesture -> keyword to reach L2
ALERT_TIMEOUT_S = 30.0    # back to IDLE after inactivity
HELP_CONFIDENCE_MIN = 0.50  # matches HUO classifier's historical default


class FusionStateMachine(object):

    def __init__(self, log_path=None, clock=None):
        # The production wall clock remains unchanged. Tests can supply a
        # callable using the same time base as message emission timestamps.
        self._clock = clock if clock is not None else time.time
        self.state = STATE_IDLE
        self.last_depth_level = None   # "FAR"|"MIDDLE"|"NEAR"|... or None
        self.last_near_score = None
        self.last_region = None        # SONG's 3x3 frame region
        self._last_region_ts = None     # emission time of that fall location
        self.alert_ts = None           # moment we entered L1
        self.last_activity_ts = self._clock()
        self._reset_ts = None          # oldest fall/help allowed after reset
        self.log_path = log_path

    # ------------------------------------------------------------------
    def update(self, msg):
        """
        Process one message (common.messages format). Returns None or an
        alert dict with level, sentence, depth_level and creation timestamp.

        Receipt time expires an inactive incident before queued data is
        processed. Message timestamps determine the fall-to-help interval.
        The UI should observe state changes as well as tick()'s return value.
        """
        if msg.get("type") != TYPE_EVENT:
            return None

        receipt_now = self._clock()
        self.tick(now=receipt_now)
        source = msg.get("source")
        payload = msg.get("payload", {})
        now = msg.get("ts", receipt_now)

        if (source in (SOURCE_GESTURE, SOURCE_SPEECH)
                and (not isinstance(now, Real) or isinstance(now, bool)
                     or not math.isfinite(now))):
            # Reject malformed input before comparisons or storing activity.
            # In particular, infinity/NaN could prevent inactivity expiry.
            return None

        if (source in (SOURCE_GESTURE, SOURCE_SPEECH)
                and self._reset_ts is not None and now <= self._reset_ts):
            # Already queued events from the cleared incident must not
            # undo a reset or attach old speech to the next incident.
            return None

        if source == SOURCE_DEPTH:
            level = payload.get("depth_level")
            # Only remember a usable relative level; ignore no-target/error.
            if level in ("FAR", "MIDDLE", "NEAR"):
                self.last_depth_level = level
                self.last_near_score = payload.get("near_score")
            return None

        if source == SOURCE_GESTURE and payload.get("gesture") == "lay_down":
            if self.state != STATE_IDLE and now < self.alert_ts:
                return None  # queued detection from before this incident
            if (payload.get("region") and (self._last_region_ts is None
                                          or now >= self._last_region_ts)):
                self.last_region = payload.get("region")
                self._last_region_ts = now
            if self.state == STATE_IDLE:
                self.state = STATE_ALERT_L1
                self.alert_ts = now
                self.last_activity_ts = now
                return self._make_alert(1)
            # A repeated detection is activity, not a new incident. In
            # particular, it must not downgrade L2 or reopen the help window.
            self.last_activity_ts = max(self.last_activity_ts, now)
            return None

        if (source == SOURCE_SPEECH and self._valid_help(payload)
                and self.state != STATE_IDLE and self.alert_ts is not None
                and now >= self.alert_ts):
            self.last_activity_ts = max(self.last_activity_ts, now)
            if (self.state == STATE_ALERT_L1
                    and now - self.alert_ts <= KEYWORD_WINDOW_S):
                self.state = STATE_ALERT_L2
                return self._make_alert(2)

        return None

    @staticmethod
    def _valid_help(payload):
        confidence = payload.get("confidence")
        return (payload.get("keyword") in KEYWORDS
                and isinstance(confidence, Real)
                and not isinstance(confidence, bool)
                and math.isfinite(confidence)
                and HELP_CONFIDENCE_MIN <= confidence <= 1.0)

    # ------------------------------------------------------------------
    def tick(self, now=None):
        """Call regularly: handles the timeout back to IDLE."""
        now = self._clock() if now is None else now
        if (self.state != STATE_IDLE
                and (now - self.last_activity_ts) > ALERT_TIMEOUT_S):
            expiry_ts = self.last_activity_ts + ALERT_TIMEOUT_S
            self._clear_incident(now, cutoff_ts=expiry_ts)
            return True  # state was reset
        return False

    # ------------------------------------------------------------------
    def reset(self, now=None):
        """Manually clear this incident and its queued fall/help events."""
        now = self._clock() if now is None else now
        self._clear_incident(now, cutoff_ts=now)

    def _clear_incident(self, now, cutoff_ts):
        """Retain usable depth; discard fall location and older detections."""
        self.state = STATE_IDLE
        self.alert_ts = None
        self.last_region = None
        self._last_region_ts = None
        self.last_activity_ts = now
        self._reset_ts = cutoff_ts

    # ------------------------------------------------------------------
    def _depth_text(self):
        parts = []
        if self.last_region:
            parts.append("{} of frame".format(self.last_region))
        if self.last_depth_level is not None:
            parts.append("relative depth {}".format(self.last_depth_level))
        return ", ".join(parts) if parts else "position unknown"

    def _make_alert(self, level):
        if level == 1:
            sentence = "ALERT: person down detected ({}).".format(
                self._depth_text())
        else:
            sentence = ("EMERGENCY: person down ({}) and calling for help. "
                        "Immediate assistance required.").format(
                            self._depth_text())

        alert = {
            "level": level,
            "sentence": sentence,
            "depth_level": self.last_depth_level,
            "ts": self._clock(),
        }
        self._log(alert)
        return alert

    def _log(self, alert):
        if not self.log_path:
            return
        try:
            with open(self.log_path, "a") as f:
                f.write("{:.3f}\tL{}\t{}\n".format(
                    alert["ts"], alert["level"], alert["sentence"]))
        except IOError:
            pass
