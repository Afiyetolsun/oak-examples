import argparse


def initialize_argparser():
    """Initialize the argument parser for the script."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.description = "This example shows how to align depth to rgb camera frame and project depth map into 3D pointcloud. You can also choose to skip rgb-depth alignment and colorize the pointcloud with right mono frame."

    parser.add_argument(
        "-d",
        "--device",
        help="Optional name, DeviceID or IP of the camera to connect to.",
        required=False,
        default=None,
        type=str,
    )

    parser.add_argument(
        "-m",
        "--mono",
        help="Use mono camera instead of RGB camera.",
        action="store_true",
    )

    parser.add_argument(
        "--fps",
        help="Camera frame rate. Lower it to reduce network bandwidth.",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--size",
        help="Output resolution as WxH (e.g. 320x200). Lower to reduce bandwidth.",
        type=str,
        default="640x400",
    )

    args = parser.parse_args()

    w, h = (int(v) for v in args.size.lower().split("x"))
    args.size = (w, h)

    return parser, args
