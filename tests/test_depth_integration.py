"""SU's pose/depth packet contract using a camera-free fake processor."""
import numpy as np
import pytest

from modules.su_depth.integration import (
    DepthIntegrationAdapter, make_pose_depth_packet,
)


class FakeProcessor:
    def __init__(self):
        self.calls = []
        self.reset_count = 0

    def process(self, frame, bbox):
        self.calls.append((frame, bbox))
        return {"depth_level": "NO_TARGET" if bbox is None else "NEAR",
                "smoothed_near_score": None if bbox is None else 0.9}

    def reset(self):
        self.reset_count += 1


@pytest.fixture
def packet():
    return make_pose_depth_packet(
        frame=np.zeros((12, 16, 3), dtype=np.uint8),
        frame_id=7, captured_at=10.0, bbox=[1, 2, 8, 9],
        pose_inferred_at=10.1, pose_confidence=0.9, track_id=3)


def test_fresh_packet_preserves_frame_and_reports_handoff_metadata(packet):
    processor = FakeProcessor()
    adapter = DepthIntegrationAdapter(processor)
    result = adapter.process_packet(packet, now=10.25)
    assert processor.calls[0][0] is packet["frame"]
    assert processor.calls[0][1] == [1.0, 2.0, 8.0, 9.0]
    assert result["integration_status"] == "OK"
    assert result["depth_level"] == "NEAR"
    assert result["frame_id"] == 7
    assert result["track_id"] == 3
    assert result["pose_confidence"] == 0.9
    assert result["packet_age_ms"] == pytest.approx(250.0)
    assert result["pose_latency_ms"] == pytest.approx(100.0)
    assert not result["track_changed"]
    assert processor.reset_count == 0


def test_stale_packet_clears_target_and_does_not_infer_on_old_bbox(packet):
    processor = FakeProcessor()
    adapter = DepthIntegrationAdapter(processor, max_packet_age_ms=1000.0)
    adapter.process_packet(packet, now=10.25)
    result = adapter.process_packet(packet, now=11.001)
    assert result["integration_status"] == "STALE_PACKET"
    assert result["depth_level"] == "NO_TARGET"
    assert processor.calls[-1][1] is None
    assert processor.reset_count == 1
    packet["captured_at"] = 12.0
    packet["track_id"] = 4
    assert not adapter.process_packet(packet, now=12.0)["track_changed"]
    assert processor.reset_count == 1


def test_packet_at_age_limit_is_still_fresh(packet):
    processor = FakeProcessor()
    adapter = DepthIntegrationAdapter(processor, max_packet_age_ms=1000.0)
    assert adapter.process_packet(packet, now=11.0)["integration_status"] == "OK"
    assert processor.calls[-1][1] == packet["bbox"]


def test_no_target_forwards_none_and_forgets_previous_identity(packet):
    processor = FakeProcessor()
    adapter = DepthIntegrationAdapter(processor)
    adapter.process_packet(packet, now=10.0)
    no_target = dict(packet, bbox=None)
    assert adapter.process_packet(no_target, now=10.0)[
        "integration_status"] == "NO_TARGET"
    assert processor.calls[-1][1] is None
    packet["track_id"] = 4
    assert not adapter.process_packet(packet, now=10.0)["track_changed"]


def test_track_change_resets_smoothing_once_before_new_target(packet):
    processor = FakeProcessor()
    adapter = DepthIntegrationAdapter(processor)
    adapter.process_packet(packet, now=10.0)
    adapter.process_packet(packet, now=10.0)
    assert processor.reset_count == 0
    packet["track_id"] = 4
    assert adapter.process_packet(packet, now=10.0)["track_changed"]
    assert processor.reset_count == 1
    assert not adapter.process_packet(packet, now=10.0)["track_changed"]
    adapter.reset()
    assert processor.reset_count == 2
    packet["track_id"] = 5
    assert not adapter.process_packet(packet, now=10.0)["track_changed"]


@pytest.mark.parametrize("field, value, exception", [
    ("protocol_version", 2, ValueError),
    ("frame_id", True, ValueError),
    ("frame_id", -1, ValueError),
    ("frame", np.zeros((12, 16)), ValueError),
    ("captured_at", float("nan"), ValueError),
    ("bbox", [1, 2, float("inf"), 9], ValueError),
    ("pose_confidence", 1.1, ValueError),
])
def test_invalid_packet_is_rejected_before_processor_call(
        packet, field, value, exception):
    processor = FakeProcessor()
    adapter = DepthIntegrationAdapter(processor)
    packet[field] = value
    with pytest.raises(exception):
        adapter.process_packet(packet, now=10.0)
    assert processor.calls == []
    assert processor.reset_count == 0


def test_missing_bbox_is_rejected_instead_of_becoming_no_target(packet):
    processor = FakeProcessor()
    adapter = DepthIntegrationAdapter(processor)
    del packet["bbox"]
    with pytest.raises(ValueError, match="missing required field: bbox"):
        adapter.process_packet(packet, now=10.0)
    assert processor.calls == []
