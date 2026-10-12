from __future__ import print_function


DEFAULT_MODEL_PATH = "checkpoints/depth_anything_v2_vits.pth"
DEFAULT_INPUT_SIZE = 224
DEFAULT_EMA_ALPHA = 0.25

DEFAULT_FAR_MIDDLE_THRESHOLD = 0.5142
DEFAULT_MIDDLE_NEAR_THRESHOLD = 0.8364


def add_depth_arguments(parser):
    """
    Add shared depth-related command-line arguments to an argparse parser.
    """
    group = parser.add_argument_group("depth")

    group.add_argument(
        "--depth-model",
        default=DEFAULT_MODEL_PATH,
        help="Path to the Depth Anything V2 checkpoint",
    )

    group.add_argument(
        "--depth-input-size",
        type=int,
        default=DEFAULT_INPUT_SIZE,
        help="Depth model input size",
    )

    group.add_argument(
        "--depth-alpha",
        type=float,
        default=DEFAULT_EMA_ALPHA,
        help="EMA smoothing alpha",
    )

    group.add_argument(
        "--depth-far-middle",
        type=float,
        default=DEFAULT_FAR_MIDDLE_THRESHOLD,
        help="FAR/MIDDLE threshold",
    )

    group.add_argument(
        "--depth-middle-near",
        type=float,
        default=DEFAULT_MIDDLE_NEAR_THRESHOLD,
        help="MIDDLE/NEAR threshold",
    )

    return parser


def depth_processor_kwargs_from_args(args):
    """
    Convert parsed argparse values into DepthProcessor keyword arguments.
    """
    return {
        "model_path": args.depth_model,
        "input_size": args.depth_input_size,
        "alpha": args.depth_alpha,
        "far_middle_threshold": args.depth_far_middle,
        "middle_near_threshold": args.depth_middle_near,
    }
