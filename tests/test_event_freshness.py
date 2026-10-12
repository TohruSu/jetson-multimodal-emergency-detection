"""Fusion queue freshness regressions with a controlled wall clock.

Message emission timestamps and fusion receipt time share the wall-clock
domain. These tests intentionally do not use pose/depth monotonic metadata.
"""
import math

import pytest

from common.messages import SOURCE_GESTURE, SOURCE_SPEECH, TYPE_EVENT
from fusion.state_machine import (
    FusionStateMachine, STATE_ALERT_L1, STATE_IDLE,
)


def event(source, payload, timestamp):
    return {"source": source, "type": TYPE_EVENT,
            "payload": payload, "ts": timestamp}


def fall(timestamp, region="right-top"):
    return event(SOURCE_GESTURE, {"gesture": "lay_down", "region": region},
                 timestamp)


def help_event(timestamp):
    return event(SOURCE_SPEECH, {"keyword": "help", "confidence": 0.9},
                 timestamp)


def test_queued_fall_older_than_current_incident_cannot_change_location():
    now = [100.0]
    fusion = FusionStateMachine(clock=lambda: now[0])
    fusion.update(fall(100.0))
    now[0] = 101.0
    assert fusion.update(fall(99.0, "left-bottom")) is None
    alert = fusion.update(help_event(101.0))
    assert "right-top" in alert["sentence"]
    assert "left-bottom" not in alert["sentence"]
    assert fusion.alert_ts == 100.0
    assert fusion.last_activity_ts == 101.0


def test_out_of_order_duplicate_cannot_replace_newer_fall_location():
    now = [100.0]
    fusion = FusionStateMachine(clock=lambda: now[0])
    fusion.update(fall(100.0, "center-top"))
    now[0] = 106.0
    fusion.update(fall(105.0, "right-top"))
    fusion.update(fall(101.0, "left-bottom"))
    alert = fusion.update(help_event(106.0))
    assert "right-top" in alert["sentence"]
    assert "left-bottom" not in alert["sentence"]
    assert fusion.alert_ts == 100.0
    assert fusion.last_activity_ts == 106.0


INVALID_TIMESTAMPS = [None, "101.0", True, math.nan, math.inf, -math.inf]


@pytest.mark.parametrize("timestamp", INVALID_TIMESTAMPS)
def test_invalid_fall_timestamp_cannot_start_or_poison_an_incident(timestamp):
    fusion = FusionStateMachine(clock=lambda: 100.0)
    assert fusion.update(fall(timestamp)) is None
    assert fusion.state == STATE_IDLE
    assert fusion.alert_ts is None
    assert fusion.last_activity_ts == 100.0
    assert fusion.last_region is None
    # A usable event must still work after a malformed queued event.
    assert fusion.update(fall(100.0))["level"] == 1
    assert fusion.tick(now=131.0)


@pytest.mark.parametrize("timestamp", INVALID_TIMESTAMPS)
def test_invalid_help_timestamp_cannot_escalate_or_extend_activity(timestamp):
    # Starting at zero also demonstrates why booleans are not timestamps:
    # True would otherwise be treated as one second after this fall.
    fusion = FusionStateMachine(clock=lambda: 0.0)
    fusion.update(fall(0.0))
    assert fusion.update(help_event(timestamp)) is None
    assert fusion.state == STATE_ALERT_L1
    assert fusion.last_activity_ts == 0.0
    assert fusion.tick(now=31.0)


def test_invalid_timestamp_is_rejected_before_reset_cutoff_comparison():
    fusion = FusionStateMachine(clock=lambda: 100.0)
    fusion.reset()
    assert fusion.update(fall("99.0")) is None
    assert fusion.state == STATE_IDLE
    assert fusion.last_region is None
    assert fusion.update(fall(101.0))["level"] == 1
