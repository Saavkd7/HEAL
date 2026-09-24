"""Reverse-engineers how visibility_masks/qcar527*_bev_visibility.png was
built, since no generator script survived (see the 2026-09-17 vault note on
this). Tests the hypothesis that the mask's angular width was computed from
each car's REAL measured fisheye intrinsics (K, D), not assumed/guessed --
by independently recovering the real camera's horizontal FOV via
cv2.fisheye.undistortPoints on the image's left/right edge pixels, and
comparing it against the mask's own angular extent (decoded with HEAL's own
box_is_visible() pixel<->metre convention, opencood/utils/box_utils.py:1236,
then divided by the project's SCALE=10 to get real metres).

    python analyze_bev_visibility_mask.py
    python analyze_bev_visibility_mask.py --calib-root /path/to/calibration_Matries
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PIPELINE_ROOT = os.path.dirname(os.path.dirname(HERE))  # scripts/visualize/ -> scripts/ -> pipeline/
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(PIPELINE_ROOT)))  # -> qcar_testbed_integration/
DEFAULT_CONF = os.path.join(HERE, "conf.json")


class _Conf(dict):
    """conf.json contents. It is THE source of every default -- nothing is
    hardcoded in this script; a CLI flag only overrides a key for one run.
    A missing key is a clear error, never a silent fallback."""

    def __init__(self, path, data):
        dict.__init__(self, data)
        self.path = path

    def __missing__(self, key):
        raise SystemExit("%s has no %r key -- add it there" % (self.path, key))

    def path_of(self, key):
        """Path-valued key: relative paths resolve against the
        qcar_testbed_integration/ root (REPO_ROOT); null stays None."""
        v = self[key]
        return v if not v or os.path.isabs(v) else os.path.join(REPO_ROOT, v)


def _load_conf(path):
    if not os.path.exists(path):
        raise SystemExit("conf.json not found: %s" % path)
    with open(path) as f:
        return _Conf(path, json.load(f))


_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--conf", default=DEFAULT_CONF)
_conf = _load_conf(_pre.parse_known_args()[0].conf)

# --- CLI flags, see qcar_dataset/pipeline/README.md's "Flag convention" ---
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument(
    "--conf", default=DEFAULT_CONF,
    help="conf.json to read defaults from (default: %s)" % DEFAULT_CONF)
_parser.add_argument(
    "--calib-root", default=_conf.path_of("calib_root"),
    help="Root of calibration_Matries/ (default: conf.json's calib_root)")
_parser.add_argument(
    "--masks-root", default=_conf.path_of("masks_root"),
    help="Root of visibility_masks/ (default: conf.json's masks_root)")
_parser.add_argument(
    "--cars", default=",".join(_conf["cars"]),
    help="Comma-separated car numbers to analyze (default: conf.json's cars)")
if __name__ == "__main__" and any(a in ("-h", "--help") for a in sys.argv[1:]):
    _parser.print_help()
    raise SystemExit(0)
_args, _ = _parser.parse_known_args()

SCALE = 10.0
_cars = [c.strip() for c in _args.cars.split(",") if c.strip()]
CARS = {c: os.path.join(_args.calib_root, "qcar%s" % c, "latest_front", "qcar%s_front_intrinsics_verified.npz" % c)
        for c in _cars}
MASKS = {c: os.path.join(_args.masks_root, "qcar%s_bev_visibility.png" % c) for c in _cars}


def real_fov_from_intrinsics(npz_path):
    d = np.load(npz_path)
    K, D = d["front_K"].astype(np.float64), d["front_D"].astype(np.float64)
    w, h = d["image_size"] if "image_size" in d else (640, 480)
    pts = np.array([[[0, h / 2]], [[w, h / 2]]], dtype=np.float64)
    und = cv2.fisheye.undistortPoints(pts, K, D)
    left_deg = np.degrees(np.arctan(np.linalg.norm(und[0, 0])))
    right_deg = np.degrees(np.arctan(np.linalg.norm(und[1, 0])))
    return left_deg, right_deg


def mask_angular_extent(mask_path):
    img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    ys, xs = np.nonzero(img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    # HEAL's own box_is_visible() convention (opencood/utils/box_utils.py:1236):
    # py = 127 - int(x/0.39), px = 127 + int(y/0.39) -- inverted here, then /SCALE for real metres
    real_x = (127 - ys) * 0.39 / SCALE
    real_y = (xs - 127) * 0.39 / SCALE
    dist = np.hypot(real_x, real_y)
    ang = np.degrees(np.arctan2(real_y, real_x))
    return ang.min(), ang.max(), dist.max(), np.percentile(dist, 90)


if __name__ == "__main__":
    print("%8s %22s %22s %14s" % ("car", "real FOV (K,D)", "mask angular extent", "mask range"))
    for car in CARS:
        left_deg, right_deg = real_fov_from_intrinsics(CARS[car])
        amin, amax, dmax, d90 = mask_angular_extent(MASKS[car])
        print("%8s  L=%6.1f R=%6.1f deg      L=%6.1f R=%6.1f deg     max=%.2fm p90=%.2fm"
              % (car, left_deg, right_deg, amin, amax, dmax, d90))
    print("\ncav_lidar_range in postprocess (opencood/hypes_yaml/.../camera_attfuse_coop.yaml): 51.2 units")
    print("51.2 / SCALE(10) = %.2f m -- compare against the mask's max range above" % (51.2 / SCALE))
