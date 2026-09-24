"""Build PretrainFront/: single-agent (qcar-52775 only) front-camera dataset
from the 101-scenario collision corpus, for the encoder-pretraining stage
("teach the weights to recognize a QCar" before the cooperation stage).

Reuses, byte-for-byte, the same schema and constants as
build_smoketest_dataset.py (x10 horizontal scale only, real qcar-52775
K/D + manual mount extrinsic, same-K fisheye undistort) -- no new geometry
invented. The only real difference: one agent per scenario folder instead
of two (HEAL's loader only requires len(cav_list) > 0, verified in
opv2v_basedataset.py), and 101 independent scenario folders instead of 1,
split at the SCENARIO level (never a frame cut) into train/validate.

Requires pretrain_labels.json (build_pretrain_labels.py) to already exist.
"""
import argparse
import json
import math
import os
import random
import shutil
import sys

import cv2
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PIPELINE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))  # scripts/build/<category>/ -> build/ -> scripts/ -> pipeline/
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(PIPELINE_ROOT)))  # -> qcar_testbed_integration/
DEFAULT_CONF = os.path.join(HERE, "conf.json")


def _load_conf(path):
    """conf.json next to this script (shared by the three build_pretrain_*.py)
    is THE source of every default below -- nothing is hardcoded here. A CLI
    flag only overrides the matching key for that one run. Relative paths in
    it are resolved against the qcar_testbed_integration/ root (REPO_ROOT)."""
    if not os.path.exists(path):
        raise SystemExit("conf.json not found: %s" % path)
    with open(path) as f:
        return json.load(f)


def _path_from_conf(key):
    """conf.json path value; relative ones are resolved against REPO_ROOT."""
    v = _conf.get(key)
    return v if not v or os.path.isabs(str(v)) else os.path.join(REPO_ROOT, v)


_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--conf", default=DEFAULT_CONF)
_conf_path = _pre.parse_known_args()[0].conf
_conf = _load_conf(_conf_path)

# --- CLI flags, see qcar_dataset/pipeline/README.md's "Flag convention" ---
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument(
    "--conf", default=DEFAULT_CONF,
    help="Path to the conf.json to read defaults from (default: %s)" % DEFAULT_CONF)
_parser.add_argument(
    "--src-root", default=_path_from_conf("src_root"),
    help="Root of the 101-scenario collision corpus, external media (default: conf.json's src_root)")
_parser.add_argument(
    "--calib-root", default=_path_from_conf("calib_root"),
    help="Root of calibration_Matries/ (default: conf.json's calib_root)")
_parser.add_argument(
    "--out-dir", default=_path_from_conf("out_dir"),
    help="Where to write PretrainFront/ (default: conf.json's out_dir)")
_parser.add_argument(
    "--labels-json", default=_path_from_conf("labels_json"),
    help="pretrain_labels.json path (default: conf.json's labels_json)")
_parser.add_argument(
    "--ego-vis-mask", default=_path_from_conf("ego_vis_mask"),
    help="Ego car's BEV visibility mask png (default: conf.json's ego_vis_mask)")
# add_help=False means -h/--help would otherwise be silently ignored (and
# this DESTRUCTIVE script -- main() does shutil.rmtree(DST) then rebuilds --
# would run for real). Handle it manually, only when run as a script.
if __name__ == "__main__" and any(a in ("-h", "--help") for a in sys.argv[1:]):
    _parser.print_help()
    raise SystemExit(0)
_args, _ = _parser.parse_known_args()
_missing = [k for k in ("src_root", "calib_root", "out_dir", "labels_json", "ego_vis_mask")
            if getattr(_args, k) is None]
if _missing:
    raise SystemExit("No value for %s: set it in %s or pass the flag."
                     % (", ".join("--" + k.replace("_", "-") for k in _missing), _conf_path))

SRC_ROOT = _args.src_root
LABELS_PATH = _args.labels_json
DST = _args.out_dir
CALIB = _args.calib_root
# Same static visibility mask reused for every frame (function of the fixed
# camera mount, not of the trajectory) -- same file already verified for
# SmokeTestFront_dev, just linked under each frame's expected name.
VIS_MASK_SRC = _args.ego_vis_mask
EGO_ID = "1"  # qcar-52775, same convention as SmokeTestFront
SCALE = 10.0
QCAR_L, QCAR_W, QCAR_H = 0.425, 0.192, 0.190
FRONT_MOUNT_XYZ = (0.1930, 0.0, 0.0953)  # manual value, same as build_smoketest_dataset.py
VAL_FRACTION = 0.15
SEED = 42

calib = np.load(os.path.join(CALIB, "qcar52775/latest_front",
                              "qcar52775_front_intrinsics_verified.npz"))
K_REAL = calib["front_K"].astype(np.float64)
D_REAL = calib["front_D"].astype(np.float64)


def undistort_front(src_png: str, dst_png: str) -> None:
    im = cv2.imread(src_png)
    out = cv2.fisheye.undistortImage(im, K_REAL, D_REAL, Knew=K_REAL)
    cv2.imwrite(dst_png, out)


def main() -> None:
    labels = json.load(open(LABELS_PATH))
    scenario_names = sorted(labels["scenarios"].keys())
    rng = random.Random(SEED)
    rng.shuffle(scenario_names)
    n_val = max(1, int(round(len(scenario_names) * VAL_FRACTION)))
    val_scenarios = set(scenario_names[:n_val])
    train_scenarios = set(scenario_names[n_val:])

    if os.path.isdir(DST):
        shutil.rmtree(DST)

    n_frames = n_pos = n_neg = 0
    cam_cords = [FRONT_MOUNT_XYZ[0] * SCALE, FRONT_MOUNT_XYZ[1] * SCALE,
                 FRONT_MOUNT_XYZ[2], 0.0, 0.0, 0.0]
    lidar_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    for si, scen_name in enumerate(scenario_names, 1):
        split = "validate" if scen_name in val_scenarios else "train"
        scen_labels = labels["scenarios"][scen_name]["labels"]
        src_scen_dir = os.path.join(SRC_ROOT, scen_name)
        dst_dir = os.path.join(DST, split, scen_name, EGO_ID)
        os.makedirs(dst_dir, exist_ok=True)

        for entry in scen_labels:
            fid = entry["frame_id"]
            src_png = os.path.join(src_scen_dir, fid, "front.png")
            dst_png = os.path.join(dst_dir, f"{fid}_camera0.png")
            undistort_front(src_png, dst_png)
            dst_vis = os.path.join(dst_dir, f"{fid}_bev_visibility.png")
            try:
                os.link(VIS_MASK_SRC, dst_vis)
            except OSError:
                shutil.copy2(VIS_MASK_SRC, dst_vis)
            n_frames += 1

            vehicles = {}
            if entry.get("camera_visible"):
                x, y, z, l, w, h, yaw_deg = entry["box_xyzlwh_yaw_deg"]
                vehicles[1] = {
                    "location": [x, y, 0.0],
                    "center": [0.0, 0.0, QCAR_H / 2.0],
                    "extent": [l / 2.0, w / 2.0, QCAR_H / 2.0],
                    "angle": [0.0, yaw_deg, 0.0],
                }
                n_pos += 1
            else:
                n_neg += 1

            params = {
                "lidar_pose": lidar_pose, "lidar_pose_clean": lidar_pose,
                "true_ego_pos": lidar_pose, "ego_speed": 0.0,
                "vehicles": vehicles,
                "qcar_provenance": {
                    "car": "qcar-52775 (192.168.1.198)",
                    "split": split, "cameras": "FRONT ONLY, single agent (pretrain stage)",
                    "source_scenario": scen_name, "source_frame": fid,
                    "pose_source": ("Vicon target pose + real qcar-52775 visibility mask "
                                    "(camera_visible check, not just Vicon-valid)"),
                    "world_scale": "x10 horizontal only, same convention as SmokeTestFront",
                    "camera0_front": "REAL calibration, offline-undistorted (same-K fisheye)",
                },
                "camera0": {
                    "cords": cam_cords, "extrinsic": np.eye(4).tolist(),
                    "intrinsic": K_REAL.tolist(),
                },
            }
            with open(os.path.join(dst_dir, f"{fid}.yaml"), "w") as f:
                yaml.safe_dump(params, f, default_flow_style=None, sort_keys=False)

        if si % 20 == 0 or si == len(scenario_names):
            print(f"[{si}/{len(scenario_names)}] scenarios written, "
                  f"{n_frames} frames so far ({n_pos} pos / {n_neg} neg)", flush=True)

    manifest = {
        "built": "2026-09-13", "purpose": "encoder pretraining -- recognize qcar, single agent",
        "ego_car": "qcar-52775 (192.168.1.198)",
        "scenarios_train": len(train_scenarios), "scenarios_validate": len(val_scenarios),
        "split_level": "scenario (never a frame cut)",
        "frames_total": n_frames, "frames_positive": n_pos, "frames_negative": n_neg,
        "world_scale": "x10 horizontal only",
        "requires": "qcar/patches/patch_1cam_loader.py monkeypatch before build_dataset()",
    }
    json.dump(manifest, open(os.path.join(DST, "manifest.json"), "w"), indent=2)
    print("\n" + json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
