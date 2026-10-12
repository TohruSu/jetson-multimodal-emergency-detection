from .config import (
    DEFAULT_EMA_ALPHA,
    DEFAULT_FAR_MIDDLE_THRESHOLD,
    DEFAULT_INPUT_SIZE,
    DEFAULT_MIDDLE_NEAR_THRESHOLD,
    DEFAULT_MODEL_PATH,
    add_depth_arguments,
    depth_processor_kwargs_from_args,
)
from .integration import (
    DEPTH_PACKET_VERSION,
    DepthIntegrationAdapter,
    make_pose_depth_packet,
    put_latest,
)
# DepthEstimator pulls the external depth_anything_v2 repo (Jetson
# only). Import it lazily so pure-logic use/tests work without it.
def __getattr__(name):
    if name == 'DepthEstimator':
        from .depth_estimator import DepthEstimator
        return DepthEstimator
    raise AttributeError(name)
from .depth_processor import DepthProcessor

__all__ = [
    "DEFAULT_MODEL_PATH",
    "DEFAULT_INPUT_SIZE",
    "DEFAULT_EMA_ALPHA",
    "DEFAULT_FAR_MIDDLE_THRESHOLD",
    "DEFAULT_MIDDLE_NEAR_THRESHOLD",
    "add_depth_arguments",
    "depth_processor_kwargs_from_args",
    "DEPTH_PACKET_VERSION",
    "DepthIntegrationAdapter",
    "make_pose_depth_packet",
    "put_latest",
    "DepthEstimator",
    "DepthProcessor",
]
