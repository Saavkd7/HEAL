"""Regenerate <pipeline>/shared/visibility_masks/qcar<NNNNN>_bev_visibility.png --
the per-car, egocentric BEV visibility cone HEAL's own box_is_visible()
consults at training/inference time (opencood/utils/box_utils.py:1236).

The ORIGINAL generator script that made the masks currently on disk did not
survive (see the research note "2026-09-17 1815 - 09 ANALISIS -- Como
se Construyo la Mascara de Visibilidad, Reconstruido desde la Geometria" for
the full reverse-engineering trail). That analysis, by measuring the existing
masks and independently recomputing a real camera FOV from calibration,
established what the mask actually encodes:

  - A 256x256 grayscale cone, in the CAV'S OWN body frame (not world frame --
    same file reused for every frame of that car, since the camera is
    rigidly mounted and never moves relative to the car body). Pixel<->metric
    convention is HEAL's own (box_utils.py:1236 docstring):

        (0,0)------------px
        |        ^ x      |
        |        |        |     x = forward, y = right (vehicle body frame)
        |        o---> y  |     py = 127 - x/0.39, px = 127 + y/0.39
        |                 |
        py-----------------(256,256)

    x, y there are in HEAL's own internal coordinate system for this
    project, i.e. REAL metres * SCALE(10) -- the same x10 scale
    build_coop_train_val_dataset.py applies to every position it writes.

  - Angular width = the car's REAL measured horizontal FOV, from its own
    verified fisheye intrinsics (K, D) -- confirmed because the two cars'
    masks have DIFFERENT angular widths (~60 deg vs ~62 deg) that each match
    that car's own K/D-derived FOV to within ~1 deg, ruling out a shared
    generic placeholder.

  - Range = postprocess.anchor_args.cav_lidar_range from the training yaml
    (51.2 units) / SCALE(10) = 5.12m real -- NOT an optical property of the
    camera, a design choice inherited from HEAL's generic OPV2V detection
    range.

This script reproduces that construction from first principles (real K/D +
the real cav_lidar_range read straight out of the training yaml), so a new
car's mask can be generated the same way once it has real verified
intrinsics -- no more hand-waiting for a lost script. It does NOT touch the
masks currently in shared/visibility_masks/ unless you pass --overwrite.

Usage:
    python build_visibility_mask.py                       # dry validation: compares
                                                            #   against the masks already on
                                                            #   disk for every calibrated car,
                                                            #   writes nothing
    python build_visibility_mask.py --overwrite            # regenerates + overwrites
    python build_visibility_mask.py --cars 52775            # only this car
    python build_visibility_mask.py --out-dir /tmp/masks    # write elsewhere instead
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PIPELINE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))  # scripts/build/<category>/ -> build/ -> scripts/ -> pipeline/
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(PIPELINE_ROOT)))  # -> qcar_testbed_integration/
DEFAULT_CONF = os.path.join(HERE, "conf.json")


def _load_conf(path):
    """conf.json next to this script is THE source of every default below --
    nothing is hardcoded here. A CLI flag only overrides the matching key for
    that one run. Relative paths in it are resolved against the
    qcar_testbed_integration/ root (REPO_ROOT)."""
    if not os.path.exists(path):
        raise SystemExit("conf.json not found: %s" % path)
    with open(path) as f:
        return json.load(f)


def _path_from_conf(key):
    """conf.json path value; relative ones are resolved against REPO_ROOT."""
    v = _conf.get(key)
    return v if not v or os.path.isabs(v) else os.path.normpath(os.path.join(REPO_ROOT, v))


_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--conf", default=DEFAULT_CONF)
_conf_path = _pre.parse_known_args()[0].conf
_conf = _load_conf(_conf_path)

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument(
    "--conf", default=DEFAULT_CONF,
    help="Path to the conf.json to read defaults from (default: %s)" % DEFAULT_CONF)
_parser.add_argument(
    "--calib-root", default=_path_from_conf("calib_root"),
    help="Root of calibration_Matries/ (default: conf.json's calib_root)")
_parser.add_argument(
    "--out-dir", default=_path_from_conf("out_dir"),
    help="Where to write <car>_bev_visibility.png (default: conf.json's out_dir)")
_parser.add_argument(
    "--cars", default=_conf["cars"],
    help="Comma-separated car names (e.g. 52775,52776) to build; default: every "
         "qcar<NNNNN>/latest_front/*_front_intrinsics_verified.npz found under --calib-root "
         "(or conf.json's cars)")
_parser.add_argument(
    "--coop-yaml", default=_path_from_conf("coop_yaml"),
    help="Training yaml to read postprocess.anchor_args.cav_lidar_range from "
         "(default: conf.json's coop_yaml)")
_parser.add_argument(
    "--overwrite", action="store_true", default=_conf["overwrite"],
    help="Actually write into --out-dir. Default: dry run -- compute masks in memory and, "
         "for any car whose mask already exists on disk, report agreement against it, "
         "writing nothing.")
if __name__ == "__main__" and any(a in ("-h", "--help") for a in sys.argv[1:]):
    _parser.print_help()
    raise SystemExit(0)
_args, _ = _parser.parse_known_args()
_missing = [k for k in ("calib_root", "out_dir", "coop_yaml")
            if getattr(_args, k) is None]
if _missing:
    raise SystemExit("No value for %s: set it in %s or pass the flag."
                     % (", ".join("--" + k.replace("_", "-") for k in _missing), _conf_path))

SCALE = 10.0
IMG_W, IMG_H = 640, 480
MASK_SIZE = 256
PX_PER_UNIT = 0.39  # HEAL's own box_is_visible() pixel<->metric factor (box_utils.py:1236)


def discover_cars(calib_root, wanted):
    found = {}
    for npz_path in sorted(glob.glob(os.path.join(calib_root, "qcar*", "latest_front", "*_front_intrinsics_verified.npz"))):
        car_dir = os.path.basename(os.path.dirname(os.path.dirname(npz_path)))  # "qcar52775"
        name = car_dir[len("qcar"):] if car_dir.startswith("qcar") else car_dir
        if wanted is not None and name not in wanted:
            continue
        found[name] = npz_path
    return found


def real_fov_from_intrinsics(K, D, w=IMG_W, h=IMG_H):
    """Real horizontal half-angles (deg) to the left/right of the optical
    axis, recovered from the car's own verified fisheye K/D by undistorting
    the real image's left/right edge pixels -- same method used to first
    confirm the mask's angular width matches real calibration (see the
    module docstring's referenced analysis)."""
    pts = np.array([[[0, h / 2.0]], [[w, h / 2.0]]], dtype=np.float64)
    und = cv2.fisheye.undistortPoints(pts, K, D)
    left_deg = np.degrees(np.arctan(np.linalg.norm(und[0, 0])))
    right_deg = np.degrees(np.arctan(np.linalg.norm(und[1, 0])))
    return left_deg, right_deg


def read_cav_lidar_range(coop_yaml_path):
    with open(coop_yaml_path) as f:
        doc = yaml.safe_load(f)
    rng = doc["cav_lidar_range"]  # [-51.2, -51.2, -3, 51.2, 51.2, 1]
    return float(rng[3])  # +x/+y bound, matches -x/-y by symmetry in this project's config


def build_mask(left_deg, right_deg, max_range_scaled):
    """256x256 uint8 cone: 255 where a point at that pixel's real (scaled)
    body-frame position is both within [-left_deg, +right_deg] of the
    forward axis AND within max_range_scaled of the origin -- HEAL's own
    box_is_visible() pixel<->metric convention (box_utils.py:1236)."""
    py, px = np.meshgrid(np.arange(MASK_SIZE), np.arange(MASK_SIZE), indexing="ij")
    x = (127 - py) * PX_PER_UNIT  # forward
    y = (px - 127) * PX_PER_UNIT  # right
    ang = np.degrees(np.arctan2(y, x))
    rng = np.hypot(x, y)
    within = (ang >= -left_deg) & (ang <= right_deg) & (rng <= max_range_scaled)
    return (within.astype(np.uint8)) * 255


def compare(new_mask, existing_path):
    old = cv2.imread(existing_path, cv2.IMREAD_UNCHANGED)
    if old is None:
        return None
    old_bin = (old > 0).astype(np.uint8)
    new_bin = (new_mask > 0).astype(np.uint8)
    inter = np.logical_and(old_bin, new_bin).sum()
    union = np.logical_or(old_bin, new_bin).sum()
    iou = inter / union if union else 1.0
    return iou, int(old_bin.sum()), int(new_bin.sum())


def main():
    wanted = set(c.strip() for c in _args.cars.split(",")) if _args.cars else None
    cars = discover_cars(_args.calib_root, wanted)
    if not cars:
        print("No calibrated car found under %s (looked for qcar*/latest_front/*_front_intrinsics_verified.npz)"
              % _args.calib_root)
        return

    max_range_real = read_cav_lidar_range(_args.coop_yaml)
    max_range_scaled = max_range_real * SCALE
    print("cav_lidar_range (from %s): %.1f -> max range %.2fm real, %.1f scaled units"
          % (os.path.basename(_args.coop_yaml), max_range_real, max_range_real / SCALE, max_range_scaled))

    os.makedirs(_args.out_dir, exist_ok=True) if _args.overwrite else None

    for name, npz_path in cars.items():
        d = np.load(npz_path)
        K, D = d["front_K"].astype(np.float64), d["front_D"].astype(np.float64)
        w, h = (d["image_size"] if "image_size" in d else (IMG_W, IMG_H))
        left_deg, right_deg = real_fov_from_intrinsics(K, D, w, h)
        mask = build_mask(left_deg, right_deg, max_range_scaled)

        out_path = os.path.join(_args.out_dir, "qcar%s_bev_visibility.png" % name)
        print("qcar%s: FOV L=%.1f R=%.1f deg, range %.2fm real" % (name, left_deg, right_deg, max_range_real / SCALE))

        if os.path.exists(out_path):
            cmp = compare(mask, out_path)
            if cmp:
                iou, n_old, n_new = cmp
                print("  vs existing %s: IoU=%.3f (existing %d px, regenerated %d px)"
                      % (out_path, iou, n_old, n_new))

        if _args.overwrite:
            cv2.imwrite(out_path, mask)
            print("  wrote", out_path)
        else:
            print("  (dry run, not written -- pass --overwrite to write %s)" % out_path)


if __name__ == "__main__":
    main()
