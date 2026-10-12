from __future__ import print_function

import time
from queue import Empty, Full

import numpy as np


DEPTH_PACKET_VERSION = 1


def make_pose_depth_packet(
    frame,
    frame_id,
    captured_at,
    bbox,
    pose_inferred_at=None,
    pose_confidence=None,
    track_id=None,
):
    """
    Create the packet sent from the Pose Consumer to the Depth Consumer.

    frame:
        OpenCV BGR numpy.ndarray. The array is not copied.

    frame_id:
        Monotonically increasing integer assigned by the Webcam Producer.

    captured_at:
        time.monotonic() timestamp assigned when the frame was captured.

    bbox:
        [x1, y1, x2, y2] in pixel coordinates, or None when no person
        is detected.

    pose_inferred_at:
        time.monotonic() timestamp recorded after pose inference.

    pose_confidence:
        Optional value from 0.0 to 1.0.

    track_id:
        Optional person identity supplied by the Pose module.
    """
    if pose_inferred_at is None:
        pose_inferred_at = time.monotonic()

    return {
        "protocol_version": DEPTH_PACKET_VERSION,
        "frame": frame,
        "frame_id": frame_id,
        "captured_at": captured_at,
        "bbox": bbox,
        "pose_inferred_at": pose_inferred_at,
        "pose_confidence": pose_confidence,
        "track_id": track_id,
    }


def put_latest(queue_object, item):
    """
    Put one item into a bounded queue without allowing backlog.

    If the queue is full, the previous item is discarded and replaced by
    the newest item.

    Returns:
        True if an old item was dropped.
        False if the queue was initially empty.
    """
    try:
        queue_object.put_nowait(item)
        return False
    except Full:
        pass

    dropped = False

    try:
        queue_object.get_nowait()
        dropped = True
    except Empty:
        pass

    queue_object.put_nowait(item)
    return dropped


class DepthIntegrationAdapter(object):
    """
    Validate Pose output packets and call a camera-free DepthProcessor.

    The adapter prevents stale Pose bbox data from being used and resets
    the EMA when the tracked person changes.
    """

    STATUS_OK = "OK"
    STATUS_NO_TARGET = "NO_TARGET"
    STATUS_STALE_PACKET = "STALE_PACKET"

    def __init__(
        self,
        processor,
        max_packet_age_ms=1000.0,
    ):
        if processor is None:
            raise ValueError("processor must not be None")

        max_packet_age_ms = float(max_packet_age_ms)

        if (
            not np.isfinite(max_packet_age_ms)
            or max_packet_age_ms <= 0.0
        ):
            raise ValueError(
                "max_packet_age_ms must be a positive finite number"
            )

        self.processor = processor
        self.max_packet_age_ms = max_packet_age_ms
        self._active_track_id = None

    @staticmethod
    def _validate_timestamp(name, value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError("{} must be numeric".format(name))

        if not np.isfinite(value):
            raise ValueError("{} must be finite".format(name))

        return value

    @staticmethod
    def _validate_frame_id(frame_id):
        if isinstance(frame_id, bool):
            raise ValueError("frame_id must be an integer")

        try:
            normalized = int(frame_id)
        except (TypeError, ValueError):
            raise ValueError("frame_id must be an integer")

        if normalized != frame_id:
            raise ValueError("frame_id must be an integer")

        if normalized < 0:
            raise ValueError("frame_id must be non-negative")

        return normalized

    @staticmethod
    def _validate_frame(frame):
        if not isinstance(frame, np.ndarray):
            raise TypeError("packet frame must be a numpy.ndarray")

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                "packet frame must have shape H x W x 3"
            )

        if frame.size == 0:
            raise ValueError("packet frame must not be empty")

        return frame

    @staticmethod
    def _validate_bbox(bbox):
        if bbox is None:
            return None

        if not isinstance(bbox, (list, tuple, np.ndarray)):
            raise TypeError(
                "packet bbox must be a list, tuple, or numpy.ndarray"
            )

        if len(bbox) != 4:
            raise ValueError(
                "packet bbox must be [x1, y1, x2, y2]"
            )

        normalized = []

        for value in bbox:
            try:
                value = float(value)
            except (TypeError, ValueError):
                raise ValueError(
                    "packet bbox coordinates must be numeric"
                )

            if not np.isfinite(value):
                raise ValueError(
                    "packet bbox coordinates must be finite"
                )

            normalized.append(value)

        return normalized

    @staticmethod
    def _validate_confidence(confidence):
        if confidence is None:
            return None

        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            raise ValueError(
                "pose_confidence must be numeric or None"
            )

        if (
            not np.isfinite(confidence)
            or confidence < 0.0
            or confidence > 1.0
        ):
            raise ValueError(
                "pose_confidence must be between 0.0 and 1.0"
            )

        return confidence

    def _validate_packet(self, packet):
        if not isinstance(packet, dict):
            raise TypeError("packet must be a dictionary")

        version = packet.get("protocol_version")

        if version != DEPTH_PACKET_VERSION:
            raise ValueError(
                "unsupported protocol_version: {}".format(version)
            )

        required_fields = (
            "frame",
            "frame_id",
            "captured_at",
            "bbox",
        )

        for field in required_fields:
            if field not in packet:
                raise ValueError(
                    "packet is missing required field: {}".format(
                        field
                    )
                )

        frame = self._validate_frame(packet["frame"])
        frame_id = self._validate_frame_id(packet["frame_id"])

        captured_at = self._validate_timestamp(
            "captured_at",
            packet["captured_at"],
        )

        bbox = self._validate_bbox(packet["bbox"])

        pose_inferred_at = packet.get("pose_inferred_at")

        if pose_inferred_at is not None:
            pose_inferred_at = self._validate_timestamp(
                "pose_inferred_at",
                pose_inferred_at,
            )

        pose_confidence = self._validate_confidence(
            packet.get("pose_confidence")
        )

        return {
            "frame": frame,
            "frame_id": frame_id,
            "captured_at": captured_at,
            "bbox": bbox,
            "pose_inferred_at": pose_inferred_at,
            "pose_confidence": pose_confidence,
            "track_id": packet.get("track_id"),
        }

    def process_packet(self, packet, now=None):
        """
        Process one Pose output packet.

        Stale packets and packets without a target do not run depth
        inference. They call process(frame, None), which clears the
        DepthProcessor EMA state.
        """
        packet = self._validate_packet(packet)

        if now is None:
            now = time.monotonic()
        else:
            now = self._validate_timestamp("now", now)

        packet_age_ms = max(
            0.0,
            (now - packet["captured_at"]) * 1000.0,
        )

        pose_latency_ms = None

        if packet["pose_inferred_at"] is not None:
            pose_latency_ms = max(
                0.0,
                (
                    packet["pose_inferred_at"]
                    - packet["captured_at"]
                )
                * 1000.0,
            )

        track_changed = False

        if packet_age_ms > self.max_packet_age_ms:
            self._active_track_id = None
            self.processor.reset()

            result = self.processor.process(
                packet["frame"],
                None,
            )

            integration_status = self.STATUS_STALE_PACKET

        elif packet["bbox"] is None:
            self._active_track_id = None

            result = self.processor.process(
                packet["frame"],
                None,
            )

            integration_status = self.STATUS_NO_TARGET

        else:
            track_id = packet["track_id"]

            if (
                track_id is not None
                and self._active_track_id is not None
                and track_id != self._active_track_id
            ):
                self.processor.reset()
                track_changed = True

            result = self.processor.process(
                packet["frame"],
                packet["bbox"],
            )

            if track_id is not None:
                self._active_track_id = track_id

            integration_status = self.STATUS_OK

        result = dict(result)

        result.update(
            {
                "protocol_version": DEPTH_PACKET_VERSION,
                "integration_status": integration_status,
                "frame_id": packet["frame_id"],
                "captured_at": packet["captured_at"],
                "pose_inferred_at": packet[
                    "pose_inferred_at"
                ],
                "pose_confidence": packet[
                    "pose_confidence"
                ],
                "track_id": packet["track_id"],
                "track_changed": track_changed,
                "packet_age_ms": float(packet_age_ms),
                "pose_latency_ms": pose_latency_ms,
            }
        )

        return result

    def reset(self):
        self._active_track_id = None
        self.processor.reset()
