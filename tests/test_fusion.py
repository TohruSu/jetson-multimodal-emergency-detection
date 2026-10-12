# -*- coding: utf-8 -*-
"""
Unit test for the state machine (no hardware needed).
Run from the project root:  python3 -m tests.test_fusion
"""
from common.messages import make_event, SOURCE_GESTURE, SOURCE_DEPTH, \
    SOURCE_SPEECH
from fusion.state_machine import (FusionStateMachine, STATE_IDLE,
                                  STATE_ALERT_L1, STATE_ALERT_L2)


class VirtualClock(object):
    """Use the same controllable time base for receipts and emitted events."""
    def __init__(self, now=100.0):
        self.now = now

    def __call__(self):
        return self.now


def event_at(clock, source, payload):
    msg = make_event(source, payload)
    msg["ts"] = clock()
    return msg


def test_full_scenario():
    clock = VirtualClock()
    fsm = FusionStateMachine(clock=clock)

    # 1. Depth publishes a level -> memorized, no alert
    alert = fsm.update(event_at(clock, SOURCE_DEPTH,
                                  {"depth_level": "NEAR",
                                   "near_score": 0.9, "status": "OK"}))
    assert alert is None
    assert fsm.state == STATE_IDLE

    # 2. "lay_down" gesture -> level 1 alert with region + depth
    clock.now = 101.0
    alert = fsm.update(event_at(clock, SOURCE_GESTURE,
                                  {"gesture": "lay_down",
                                   "confidence": 0.9,
                                   "region": "center-bottom"}))
    assert alert is not None and alert["level"] == 1
    assert "NEAR" in alert["sentence"]
    assert "center-bottom" in alert["sentence"]
    assert fsm.state == STATE_ALERT_L1

    # 3. "help" keyword within the 10 s window -> level 2 alert
    clock.now = 104.0
    alert = fsm.update(event_at(clock, SOURCE_SPEECH,
                                  {"keyword": "help", "confidence": 0.88}))
    assert alert is not None and alert["level"] == 2
    assert "EMERGENCY" in alert["sentence"]
    assert fsm.state == STATE_ALERT_L2

    # 4. 30 s timeout -> back to IDLE
    clock.now = 135.0
    reset = fsm.tick()
    assert reset is True
    assert fsm.state == STATE_IDLE

    print("test_full_scenario: OK")


def test_help_without_gesture_does_not_trigger_l2():
    clock = VirtualClock()
    fsm = FusionStateMachine(clock=clock)
    alert = fsm.update(event_at(clock, SOURCE_SPEECH,
                                  {"keyword": "help", "confidence": 0.9}))
    assert alert is None
    assert fsm.state == STATE_IDLE
    print("test_help_without_gesture: OK")


def test_manual_reset():
    clock = VirtualClock()
    fsm = FusionStateMachine(clock=clock)
    fsm.update(event_at(clock, SOURCE_GESTURE,
                          {"gesture": "lay_down", "confidence": 0.9}))
    assert fsm.state == STATE_ALERT_L1
    fsm.reset()
    assert fsm.state == STATE_IDLE
    print("test_manual_reset: OK")


if __name__ == "__main__":
    test_full_scenario()
    test_help_without_gesture_does_not_trigger_l2()
    test_manual_reset()
    print("All tests pass.")
