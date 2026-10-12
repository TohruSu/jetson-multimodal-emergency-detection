import time
from pathlib import Path

import cv2
import numpy as np
import torch

from depth_anything_v2.dpt import DepthAnythingV2


MODEL_CONFIGS = {
    "vits": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
    }
}


class DepthEstimator:
    """
    Depth Anything V2 Small wrapper.

    Main integration interface:
        depth_map, inference_ms = estimator.infer(frame)
        result = estimator.estimate_bbox_depth(depth_map, bbox)

    The relative model does not directly output distance in meters.
    """

    def __init__(
        self,
        model_path=None,
        input_size=350,
        device=None,
    ):
        project_root = Path(__file__).resolve().parents[1]

        if model_path is None:
            model_path = (
                project_root
                / "checkpoints"
                / "depth_anything_v2_vits.pth"
            )

        self.model_path = Path(model_path)
        self.input_size = int(input_size)

        if not self.model_path.exists():
            raise FileNotFoundError(
                "Depth model was not found: {}".format(
                    self.model_path.resolve()
                )
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        print("Loading Depth Anything V2 Small...")
        print("Model path:", self.model_path.resolve())
        print("Device:", self.device)
        print("Input size:", self.input_size)

        self.model = DepthAnythingV2(**MODEL_CONFIGS["vits"])

        # New PyTorch versions support weights_only.
        # The fallback helps compatibility with older Jetson PyTorch builds.
        try:
            state_dict = torch.load(
                str(self.model_path),
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            state_dict = torch.load(
                str(self.model_path),
                map_location="cpu",
            )

        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()

        print("Depth model loaded successfully.")

    def infer(self, frame):
        """
        Estimate relative depth from one OpenCV BGR image.

        Args:
            frame:
                H x W x 3 uint8 OpenCV BGR image.

        Returns:
            depth_map:
                H x W float32 relative depth map.

            inference_ms:
                Inference time in milliseconds.
        """
        if frame is None:
            raise ValueError("frame must not be None")

        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a NumPy array")

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                "frame must have shape H x W x 3"
            )

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        with torch.inference_mode():
            depth_map = self.model.infer_image(
                frame,
                self.input_size,
            )

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        inference_ms = (
            time.perf_counter() - start_time
        ) * 1000.0

        depth_map = np.asarray(
            depth_map,
            dtype=np.float32,
        )

        return depth_map, inference_ms

    def estimate_bbox_depth(
        self,
        depth_map,
        bbox,
        inner_ratio=0.20,
    ):
        """
        Estimate a representative depth value inside a bounding box.

        Args:
            depth_map:
                H x W relative depth map.

            bbox:
                [x1, y1, x2, y2], in original image coordinates.

            inner_ratio:
                Percentage removed from each side of the bounding box.
                This reduces background contamination around the person.

        Returns:
            Dictionary containing bbox position, raw relative depth,
            and a normalized near score.

        near_score:
            0.0 means relatively far in the current image.
            1.0 means relatively near in the current image.
        """
        if depth_map is None:
            raise ValueError("depth_map must not be None")

        if depth_map.ndim != 2:
            raise ValueError("depth_map must be H x W")

        if bbox is None or len(bbox) != 4:
            raise ValueError(
                "bbox must be [x1, y1, x2, y2]"
            )

        height, width = depth_map.shape

        x1, y1, x2, y2 = [int(round(v)) for v in bbox]

        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height))

        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                "bbox has zero or negative area after clipping"
            )

        bbox_width = x2 - x1
        bbox_height = y2 - y1

        pad_x = int(bbox_width * inner_ratio)
        pad_y = int(bbox_height * inner_ratio)

        roi_x1 = x1 + pad_x
        roi_x2 = x2 - pad_x
        roi_y1 = y1 + pad_y
        roi_y2 = y2 - pad_y

        # Fall back to the complete box when the inner area is too small.
        if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
            roi_x1, roi_y1 = x1, y1
            roi_x2, roi_y2 = x2, y2

        roi = depth_map[
            roi_y1:roi_y2,
            roi_x1:roi_x2,
        ]

        valid_roi = roi[np.isfinite(roi)]

        if valid_roi.size == 0:
            raise ValueError(
                "No valid depth value was found inside bbox"
            )

        # Remove extreme values before taking the median.
        if valid_roi.size >= 20:
            lower_roi, upper_roi = np.percentile(
                valid_roi,
                [5.0, 95.0],
            )

            trimmed_roi = valid_roi[
                (valid_roi >= lower_roi)
                & (valid_roi <= upper_roi)
            ]

            if trimmed_roi.size > 0:
                valid_roi = trimmed_roi

        raw_depth_value = float(np.median(valid_roi))

        valid_frame = depth_map[np.isfinite(depth_map)]

        if valid_frame.size == 0:
            near_score = 0.0
        else:
            frame_low, frame_high = np.percentile(
                valid_frame,
                [2.0, 98.0],
            )

            frame_range = float(frame_high - frame_low)

            if frame_range < 1e-8:
                near_score = 0.0
            else:
                near_score = (
                    raw_depth_value - float(frame_low)
                ) / frame_range

                near_score = float(
                    np.clip(near_score, 0.0, 1.0)
                )

        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        return {
            "bbox": [x1, y1, x2, y2],
            "bbox_center": [center_x, center_y],
            "roi_bbox": [
                roi_x1,
                roi_y1,
                roi_x2,
                roi_y2,
            ],
            "relative_depth_raw": raw_depth_value,
            "near_score": near_score,
        }

    @staticmethod
    def colorize_depth(depth_map):
        """
        Convert a float depth map to an OpenCV color image.
        """
        valid = depth_map[np.isfinite(depth_map)]

        if valid.size == 0:
            return np.zeros(
                (
                    depth_map.shape[0],
                    depth_map.shape[1],
                    3,
                ),
                dtype=np.uint8,
            )

        low, high = np.percentile(valid, [2.0, 98.0])
        value_range = float(high - low)

        if value_range < 1e-8:
            normalized = np.zeros_like(
                depth_map,
                dtype=np.uint8,
            )
        else:
            normalized = (
                np.clip(
                    (depth_map - low) / value_range,
                    0.0,
                    1.0,
                )
                * 255.0
            ).astype(np.uint8)

        return cv2.applyColorMap(
            normalized,
            getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET),
        )