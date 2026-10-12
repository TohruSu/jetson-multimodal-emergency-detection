from __future__ import print_function

import numpy as np

from .config import (
    DEFAULT_EMA_ALPHA as CONFIG_DEFAULT_EMA_ALPHA,
    DEFAULT_FAR_MIDDLE_THRESHOLD as CONFIG_DEFAULT_FAR_MIDDLE_THRESHOLD,
    DEFAULT_INPUT_SIZE as CONFIG_DEFAULT_INPUT_SIZE,
    DEFAULT_MIDDLE_NEAR_THRESHOLD as CONFIG_DEFAULT_MIDDLE_NEAR_THRESHOLD,
    DEFAULT_MODEL_PATH as CONFIG_DEFAULT_MODEL_PATH,
)


class DepthProcessor(object):
    """
    Stateful, camera-free depth processing interface.

    This class owns:
      - Depth Anything V2 inference
      - bbox depth extraction
      - EMA smoothing
      - FAR / MIDDLE / NEAR classification

    This class does NOT:
      - open a camera
      - read /dev/video*
      - create a GUI
      - manage producer-consumer queues

    Input frame:
      OpenCV BGR image as H x W x 3 numpy.ndarray.

    Input bbox:
      [x1, y1, x2, y2] in pixel coordinates.
    """

    DEFAULT_INPUT_SIZE = CONFIG_DEFAULT_INPUT_SIZE
    DEFAULT_ALPHA = CONFIG_DEFAULT_EMA_ALPHA
    DEFAULT_FAR_MIDDLE_THRESHOLD = (
        CONFIG_DEFAULT_FAR_MIDDLE_THRESHOLD
    )
    DEFAULT_MIDDLE_NEAR_THRESHOLD = (
        CONFIG_DEFAULT_MIDDLE_NEAR_THRESHOLD
    )

    def __init__(
        self,
        model_path=CONFIG_DEFAULT_MODEL_PATH,
        input_size=DEFAULT_INPUT_SIZE,
        alpha=DEFAULT_ALPHA,
        far_middle_threshold=DEFAULT_FAR_MIDDLE_THRESHOLD,
        middle_near_threshold=DEFAULT_MIDDLE_NEAR_THRESHOLD,
        estimator=None,
    ):
        self.input_size = int(input_size)
        self.alpha = float(alpha)
        self.far_middle_threshold = float(far_middle_threshold)
        self.middle_near_threshold = float(middle_near_threshold)

        self._validate_config()

        # estimator injection is supported so the control logic can be tested
        # without loading the model, CUDA, or a physical camera.
        if estimator is None:
            # Imported here (not at module level) so this file stays
            # usable without the Jetson-only depth_anything_v2 repo.
            from .depth_estimator import DepthEstimator
            self.estimator = DepthEstimator(
                model_path=model_path,
                input_size=self.input_size,
            )
        else:
            self.estimator = estimator

        self._smoothed_near_score = None
        self._last_depth_map = None
        self._last_result = None

    def _validate_config(self):
        if self.input_size <= 0:
            raise ValueError("input_size must be greater than 0")

        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must satisfy 0.0 < alpha <= 1.0")

        if not (
            0.0
            <= self.far_middle_threshold
            < self.middle_near_threshold
            <= 1.0
        ):
            raise ValueError(
                "thresholds must satisfy "
                "0.0 <= far_middle < middle_near <= 1.0"
            )

    @staticmethod
    def _validate_frame(frame):
        if frame is None:
            raise ValueError("frame must not be None")

        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a numpy.ndarray")

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must have shape H x W x 3")

        if frame.size == 0:
            raise ValueError("frame must not be empty")

        if frame.shape[0] <= 0 or frame.shape[1] <= 0:
            raise ValueError("frame height and width must be positive")

    @staticmethod
    def _normalize_bbox(frame, bbox):
        if bbox is None:
            return None

        if not isinstance(bbox, (list, tuple, np.ndarray)):
            raise TypeError(
                "bbox must be a list, tuple, or numpy.ndarray"
            )

        if len(bbox) != 4:
            raise ValueError("bbox must be [x1, y1, x2, y2]")

        coordinates = []

        for value in bbox:
            try:
                number = float(value)
            except (TypeError, ValueError):
                raise ValueError("bbox coordinates must be numeric")

            if not np.isfinite(number):
                raise ValueError("bbox coordinates must be finite")

            coordinates.append(int(round(number)))

        frame_height, frame_width = frame.shape[:2]
        x1, y1, x2, y2 = coordinates

        # Clip pose-module output to the current frame.
        x1 = max(0, min(x1, frame_width - 1))
        y1 = max(0, min(y1, frame_height - 1))
        x2 = max(1, min(x2, frame_width))
        y2 = max(1, min(y2, frame_height))

        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                "bbox has no positive area after frame clipping"
            )

        return [x1, y1, x2, y2]

    def classify_score(self, smoothed_near_score):
        score = float(smoothed_near_score)

        if score >= self.middle_near_threshold:
            return "NEAR"

        if score >= self.far_middle_threshold:
            return "MIDDLE"

        return "FAR"

    def _make_no_target_result(self):
        return {
            "bbox": None,
            "bbox_center": None,
            "roi_bbox": None,
            "relative_depth_raw": None,
            "near_score": None,
            "smoothed_near_score": None,
            "depth_level": "NO_TARGET",
            "inference_ms": 0.0,
            "is_metric_distance": False,
        }

    def process(self, frame, bbox):
        """
        Run depth inference for one externally supplied frame and bbox.

        Returns a lightweight dictionary for state fusion.

        near_score and smoothed_near_score are relative scores:
          0.0 = relatively far in the current frame
          1.0 = relatively near in the current frame

        They are not metres or centimetres.
        """
        self._validate_frame(frame)
        normalized_bbox = self._normalize_bbox(frame, bbox)

        # No detected person: skip the expensive model and remove stale EMA.
        if normalized_bbox is None:
            self._smoothed_near_score = None
            self._last_depth_map = None
            self._last_result = self._make_no_target_result()
            return dict(self._last_result)

        depth_map, inference_ms = self.estimator.infer(frame)

        bbox_result = self.estimator.estimate_bbox_depth(
            depth_map,
            normalized_bbox,
        )

        near_score = float(
            np.clip(float(bbox_result["near_score"]), 0.0, 1.0)
        )

        if self._smoothed_near_score is None:
            self._smoothed_near_score = near_score
        else:
            self._smoothed_near_score = (
                self.alpha * near_score
                + (1.0 - self.alpha) * self._smoothed_near_score
            )

        smoothed_score = float(self._smoothed_near_score)
        depth_level = self.classify_score(smoothed_score)

        result = dict(bbox_result)
        result.update(
            {
                "near_score": near_score,
                "smoothed_near_score": smoothed_score,
                "depth_level": depth_level,
                "inference_ms": float(inference_ms),
                "is_metric_distance": False,
            }
        )

        self._last_depth_map = depth_map
        self._last_result = result

        return dict(result)

    def warmup(self, frame):
        """
        Run one model inference without changing bbox EMA state.

        The frame must be supplied by the external Webcam Producer.
        """
        self._validate_frame(frame)

        depth_map, inference_ms = self.estimator.infer(frame)
        self._last_depth_map = depth_map

        return float(inference_ms)

    def reset(self):
        """
        Clear target-dependent state.

        Call this when the tracked person changes identity or the stream
        restarts.
        """
        self._smoothed_near_score = None
        self._last_depth_map = None
        self._last_result = None

    def get_last_result(self):
        if self._last_result is None:
            return None

        return dict(self._last_result)

    def get_last_depth_map(self, copy=False):
        if self._last_depth_map is None:
            return None

        if copy:
            return self._last_depth_map.copy()

        return self._last_depth_map

    def get_last_depth_color(self):
        if self._last_depth_map is None:
            return None

        return self.estimator.colorize_depth(self._last_depth_map)
