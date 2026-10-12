# -*- coding: utf-8 -*-
"""Fusion regressions exercised against the original archive before fixes.

These tests use only the standard library and a virtual wall clock. Run with
``python -m unittest tests.test_fusion_regressions`` or pytest.
"""
import math
import unittest
from unittest.mock import patch

from common.messages import (SOURCE_DEPTH, SOURCE_GESTURE, SOURCE_SPEECH,
                             TYPE_EVENT, TYPE_STATUS)
from fusion.state_machine import (FusionStateMachine, STATE_IDLE,
                                  STATE_ALERT_L1, STATE_ALERT_L2)


class FusionRegressionTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.0
        self.clock_patch = patch("fusion.state_machine.time.time",
                                 side_effect=lambda: self.now)
        self.clock_patch.start()
        self.addCleanup(self.clock_patch.stop)
        self.fsm = FusionStateMachine()

    def event(self, source, payload, ts):
        self.now = max(self.now, ts)
        return self.fsm.update({"source": source, "type": TYPE_EVENT,
                                "payload": payload, "ts": ts})

    def fall(self, ts=100.0, region=None):
        payload = {"gesture": "lay_down", "confidence": 0.9}
        if region is not None:
            payload["region"] = region
        return self.event(SOURCE_GESTURE, payload, ts)

    def help(self, ts=101.0, keyword="help", confidence=0.9):
        return self.event(SOURCE_SPEECH,
                          {"keyword": keyword, "confidence": confidence}, ts)

    def assert_rejected_speech(self, confidence=0.9, keyword="help"):
        self.fall()
        self.assertIsNone(self.help(keyword=keyword, confidence=confidence))
        self.assertEqual(self.fsm.state, STATE_ALERT_L1)
        self.assertEqual(self.fsm.last_activity_ts, 100.0)

    def test_unrelated_word_cannot_escalate(self):
        self.assert_rejected_speech(keyword="hello")

    def test_low_confidence_cannot_escalate(self):
        self.assert_rejected_speech(confidence=0.49)

    def test_missing_confidence_cannot_escalate(self):
        self.fall()
        self.assertIsNone(self.event(SOURCE_SPEECH, {"keyword": "help"}, 101))
        self.assertEqual(self.fsm.state, STATE_ALERT_L1)

    def test_string_confidence_is_not_a_detection_score(self):
        self.assert_rejected_speech(confidence="0.9")

    def test_boolean_confidence_is_not_a_detection_score(self):
        self.assert_rejected_speech(confidence=True)

    def test_nan_confidence_cannot_escalate(self):
        self.assert_rejected_speech(confidence=math.nan)

    def test_infinite_confidence_cannot_escalate(self):
        self.assert_rejected_speech(confidence=math.inf)

    def test_confidence_above_one_cannot_escalate(self):
        self.assert_rejected_speech(confidence=1.01)

    def test_negative_confidence_cannot_escalate(self):
        self.assert_rejected_speech(confidence=-0.1)

    def test_pre_fall_help_cannot_escalate_or_extend_activity(self):
        self.fall()
        self.assertIsNone(self.help(ts=99.0))
        self.assertEqual(self.fsm.state, STATE_ALERT_L1)
        self.assertEqual(self.fsm.last_activity_ts, 100.0)

    def test_confidence_threshold_is_inclusive(self):
        self.fall()
        self.assertEqual(self.help(confidence=0.50)["level"], 2)

    def test_aide_is_recognized_help(self):
        self.fall()
        self.assertEqual(self.help(keyword="aide")["level"], 2)

    def test_tasukete_is_recognized_help(self):
        self.fall()
        self.assertEqual(self.help(keyword="tasukete")["level"], 2)

    def test_temporal_window_includes_zero_and_ten_seconds(self):
        self.fall()
        self.assertEqual(self.help(ts=100.0)["level"], 2)
        self.fsm.reset()
        self.fall(ts=200.0)
        self.assertEqual(self.help(ts=210.0)["level"], 2)

    def test_late_valid_help_refreshes_activity_without_escalation(self):
        self.fall()
        self.assertIsNone(self.help(ts=120.0))
        self.assertEqual(self.fsm.state, STATE_ALERT_L1)
        self.assertFalse(self.fsm.tick(now=131.0))

    def test_duplicate_fall_preserves_original_escalation_window(self):
        self.fall()
        self.fall(ts=109.0)
        self.assertEqual(self.fsm.alert_ts, 100.0)
        self.assertIsNone(self.help(ts=111.0))
        self.assertEqual(self.fsm.state, STATE_ALERT_L1)

    def test_duplicate_fall_cannot_downgrade_l2(self):
        self.fall()
        self.help()
        duplicate_alert = self.fall(ts=105.0)
        self.assertEqual(self.fsm.state, STATE_ALERT_L2)
        self.assertTrue(duplicate_alert is None or duplicate_alert["level"] == 2)
        self.assertEqual(self.fsm.alert_ts, 100.0)
        self.assertEqual(self.fsm.last_activity_ts, 105.0)

    def test_out_of_order_fall_cannot_move_activity_backwards(self):
        self.fall()
        self.help(ts=102.0)
        self.fall(ts=101.0)
        self.assertEqual(self.fsm.last_activity_ts, 102.0)
        self.assertEqual(self.fsm.state, STATE_ALERT_L2)

    def test_irrelevant_speech_cannot_keep_alert_alive(self):
        self.fall()
        self.help(ts=129.0, keyword="hello")
        self.assertTrue(self.fsm.tick(now=131.0))
        self.assertEqual(self.fsm.state, STATE_IDLE)

    def test_low_confidence_speech_cannot_keep_alert_alive(self):
        self.fall()
        self.help(ts=129.0, confidence=0.49)
        self.assertTrue(self.fsm.tick(now=131.0))

    def test_event_after_timeout_cannot_revive_expired_alert(self):
        self.fall()
        self.assertIsNone(self.help(ts=131.0))
        self.assertEqual(self.fsm.state, STATE_IDLE)
        self.assertIsNone(self.fsm.alert_ts)

    def test_queued_help_uses_receipt_clock_for_expiration(self):
        self.fall()
        self.now = 131.0
        self.assertIsNone(self.help(ts=101.0))
        self.assertEqual(self.fsm.state, STATE_IDLE)

    def test_new_fall_after_timeout_starts_a_fresh_incident(self):
        self.fall(region="left-bottom")
        alert = self.fall(ts=131.0)
        self.assertEqual(alert["level"], 1)
        self.assertEqual(self.fsm.alert_ts, 131.0)
        self.assertNotIn("left-bottom", alert["sentence"])

    def test_timeout_boundary_remains_strictly_greater_than_thirty(self):
        self.fall()
        self.assertFalse(self.fsm.tick(now=130.0))
        self.assertTrue(self.fsm.tick(now=130.001))
        self.assertFalse(self.fsm.tick(now=131.0))

    def test_explicit_zero_tick_uses_zero_instead_of_wall_clock(self):
        self.now = 0.0
        self.fall(ts=0.0)
        self.now = 100.0
        self.assertFalse(self.fsm.tick(now=0.0))
        self.assertEqual(self.fsm.state, STATE_ALERT_L1)

    def test_reset_clears_fall_region_but_keeps_current_depth(self):
        self.event(SOURCE_DEPTH, {"depth_level": "NEAR", "near_score": 0.8}, 100)
        self.fall(region="right-bottom")
        self.fsm.reset()
        self.assertEqual(self.fsm.state, STATE_IDLE)
        self.assertIsNone(self.fsm.alert_ts)
        alert = self.fall(ts=101)
        self.assertNotIn("right-bottom", alert["sentence"])
        self.assertEqual(alert["depth_level"], "NEAR")

    def test_queued_pre_reset_fall_cannot_undo_manual_reset(self):
        self.fall(region="left-bottom")
        self.now = 104.0
        self.fsm.reset()
        self.assertIsNone(self.fall(ts=100.0, region="left-bottom"))
        self.assertIsNone(self.fall(ts=104.0, region="left-bottom"))
        self.assertEqual(self.fsm.state, STATE_IDLE)
        self.assertIsNone(self.fsm.alert_ts)
        self.assertIsNone(self.fsm.last_region)

    def test_queued_old_fall_cannot_reopen_timed_out_incident(self):
        self.fall()
        self.now = 140.0
        self.assertTrue(self.fsm.tick())
        self.assertIsNone(self.fall(ts=100.0))
        self.assertIsNone(self.fall(ts=130.0))
        self.assertEqual(self.fsm.state, STATE_IDLE)

    def test_fresh_fall_after_expiry_survives_queue_delay(self):
        self.fall()
        self.now = 140.0
        self.assertTrue(self.fsm.tick())
        alert = self.fall(ts=135.0)
        self.assertEqual(alert["level"], 1)
        self.assertEqual(self.fsm.state, STATE_ALERT_L1)
        self.assertEqual(self.fsm.alert_ts, 135.0)
        self.assertFalse(self.fsm.tick(now=140.0))

    def test_timeout_clears_fall_region(self):
        self.fall(region="center-bottom")
        self.assertTrue(self.fsm.tick(now=131))
        self.assertNotIn("center-bottom", self.fall(ts=132)["sentence"])

    def test_depth_and_status_do_not_extend_alert_activity(self):
        self.fall()
        self.event(SOURCE_DEPTH, {"depth_level": "FAR"}, 120)
        self.now = 129
        self.assertIsNone(self.fsm.update({"source": SOURCE_SPEECH,
                                          "type": TYPE_STATUS, "ts": 129,
                                          "payload": {"keyword": "help"}}))
        self.assertTrue(self.fsm.tick(now=131))

    def test_valid_repeated_help_keeps_l2_until_inactivity_timeout(self):
        self.fall()
        self.help(ts=101)
        self.assertIsNone(self.help(ts=120))
        self.assertEqual(self.fsm.state, STATE_ALERT_L2)
        self.assertFalse(self.fsm.tick(now=150))
        self.assertTrue(self.fsm.tick(now=150.01))


class FusionClockTests(unittest.TestCase):
    def test_injected_clock_supports_events_without_timestamp_and_reset(self):
        now = [0.0]
        fsm = FusionStateMachine(clock=lambda: now[0])
        alert = fsm.update({"source": SOURCE_GESTURE, "type": TYPE_EVENT,
                            "payload": {"gesture": "lay_down"}})
        self.assertEqual(alert["ts"], 0.0)
        self.assertEqual(fsm.alert_ts, 0.0)
        now[0] = 10.0
        alert = fsm.update({"source": SOURCE_SPEECH, "type": TYPE_EVENT,
                            "payload": {"keyword": "help", "confidence": 0.9}})
        self.assertEqual(alert["level"], 2)
        self.assertEqual(alert["ts"], 10.0)
        now[0] = 41.0
        self.assertTrue(fsm.tick())
        fsm.reset(now=0.0)
        self.assertEqual(fsm.last_activity_ts, 0.0)


if __name__ == "__main__":
    unittest.main()
