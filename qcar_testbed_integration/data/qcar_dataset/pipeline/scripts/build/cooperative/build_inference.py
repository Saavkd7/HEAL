"""Build InferenceFront/: the same real two-agent captures as
build_coop_train_val_dataset.py, but stripped down to exactly what HEAL needs to RUN
(not TRAIN) -- no ground truth, no visibility computation, no scene_gt
oracle. Real deployment/inference never knows ground truth in advance, so
this script never computes it.

Kept vs. build_coop_train_val_dataset.py:
  - load_frames / pair_frames (same real Vicon pose + timestamp pairing)
  - undistort (same real per-car fisheye K/D)
  - lidar_pose / cam_cords / intrinsic in each agent's yaml (still required
    structurally -- HEAL's get_ext_int() and the LSS geometry pipeline need
    the ego pose and camera intrinsics regardless of whether GT exists)
  - bev_visibility.png link (still required structurally -- the loader's
    add_data_extension config tries to open it unconditionally; see the
    2026-09-14 crash this project already hit and fixed once)

Dropped (training-only, meaningless for pure inference):
  - build_scene() / truly_visible_to_agent() / WALL_BLOCKED / STATIC_TARGETS
    -- all of this exists ONLY to compute correct ground-truth labels; a
    real inference run doesn't know where anything "really" is in advance.
  - vehicles{} is always {} (empty) -- matches this project's own established
    "negative frame" convention (see visualize_boxes.py: an empty vehicles{}
    is handled as a normal, valid, no-GT frame, not an error).
  - scene_gt/ -- that was an evaluation oracle to score against; nothing to
    score without ground truth.
  - No train/validate split -- inference doesn't hold out data from itself.

    python build_inference.py
    python build_inference.py --onboard-root /path/to/new/capture --out-dir /path/to/out
"""
import argparse
import json
import math
import os
import re
import shutil
import sys

import cv2
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PIPELINE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))  # scripts/build/<category>/ -> build/ -> scripts/ -> pipeline/
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(PIPELINE_ROOT)))
DEFAULT_CONF = os.path.join(HERE, "conf.json")


def _load_conf(path):
    """conf.json next to this script (shared with build_coop_train_val_dataset.py)
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
    return v if not v or os.path.isabs(v) else os.path.join(REPO_ROOT, v)


_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--conf", default=DEFAULT_CONF)
_conf_path = _pre.parse_known_args()[0].conf
_conf = _load_conf(_conf_path)

# Same flag convention as build_coop_train_val_dataset.py (see that file's own comment):
# add_help=False + parse_known_args so importing this module never steals
# -h/--help from a caller, and this destructive script (rebuilds --out-dir
# from scratch) never runs off an accidental --help typo.
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument(
    "--conf", default=DEFAULT_CONF,
    help="Path to the conf.json to read defaults from (default: %s)" % DEFAULT_CONF)
_parser.add_argument(
    "--calib-root", default=_path_from_conf("calib_root"),
    help="Root of calibration_Matries/ (default: conf.json's calib_root)")
_parser.add_argument(
    "--onboard-root", default=_path_from_conf("onboard_root"),
    help="Root holding one <ip>/dataset/ per car (default: conf.json's onboard_root)")
_parser.add_argument(
    "--out-dir", default=_path_from_conf("inference_out_dir"),
    help="Where to write the inference-ready dataset (default: conf.json's inference_out_dir)")
_parser.add_argument(
    "--vis-masks-dir", default=_path_from_conf("vis_masks_dir"),
    help="Folder with each car's <car>_bev_visibility.png (default: conf.json's vis_masks_dir)")
if __name__ == "__main__" and any(a in ("-h", "--help") for a in sys.argv[1:]):
    _parser.print_help()
    raise SystemExit(0)
_args, _ = _parser.parse_known_args()
_missing = [k for k in ("calib_root", "onboard_root", "out_dir", "vis_masks_dir")
            if getattr(_args, k) is None]
if _missing:
    raise SystemExit("No value for %s: set it in %s or pass the flag."
                     % (", ".join("--" + k.replace("_", "-") for k in _missing), _conf_path))

CALIB = _args.calib_root
ONBOARD_ROOT = _args.onboard_root
DST = _args.out_dir
VIS_MASKS = _args.vis_masks_dir

SCALE = 10.0
EGO_ID, PEER_ID = "1", "2"
PAIR_TOLERANCE_SEC = _conf["pair_tolerance_sec"]  # same real-time pairing tolerance used for training

# Physical car registry -- same identity mapping as build_coop_train_val_dataset.py's
# CAR_REGISTRY, kept as a Python constant (not conf.json) on purpose: it's
# fixed hardware identity, not a run parameter, and auto-deriving it by
# sorting IPs on disk would risk silently swapping which car is agent "1"
# vs "2" (see build_coop_train_val_dataset.py's own CAR_REGISTRY comment for why).
CAR_REGISTRY = {
    EGO_ID: {"ip": "192.168.1.198", "calib_name": "qcar52775",
             "car_label": "qcar-52775 (192.168.1.198)",
             "vis_mask_file": "qcar52775_bev_visibility.png"},
    PEER_ID: {"ip": "192.168.1.158", "calib_name": "qcar52776",
              "car_label": "qcar-52776 (192.168.1.158)",
              "vis_mask_file": "qcar52776_bev_visibility.png"},
}


def _load_cam_params():
    params = {}
    for aid, reg in CAR_REGISTRY.items():
        calib = np.load(os.path.join(
            CALIB, reg["calib_name"], "latest_front",
            "%s_front_intrinsics_verified.npz" % reg["calib_name"]))
        params[aid] = {
            "K": calib["front_K"].astype(np.float64), "D": calib["front_D"].astype(np.float64),
            "car": reg["car_label"],
            "root": os.path.join(ONBOARD_ROOT, reg["ip"], "dataset"),
            "vis_mask": os.path.join(VIS_MASKS, reg["vis_mask_file"]),
        }
    return params


CAM_PARAMS = _load_cam_params()
FRONT_MOUNT_XYZ = (0.1930, 0.0, 0.0953)  # manual value, used for cam_cords (rotation handled by the patch)

_TRAJ_INDEX_RE = re.compile(r"^converted_.+_(\d+)$")


def discover_trajectories(cav_id):
    """Scan <root>/ for every converted_*_<NN> folder and return
    {index: folder_name} -- same convention as build_coop_train_val_dataset.py, so a
    new capture (any index, any route name) is picked up automatically, no
    hardcoded route-name prefix or trajectory count."""
    root = CAM_PARAMS[cav_id]["root"]
    found = {}
    if not os.path.isdir(root):
        return found
    for name in sorted(os.listdir(root)):
        m = _TRAJ_INDEX_RE.match(name)
        if m and os.path.isdir(os.path.join(root, name)):
            found[m.group(1)] = name
    return found


def common_indices():
    """Trajectory indices present on BOTH cars, sorted."""
    a = discover_trajectories(EGO_ID)
    b = discover_trajectories(PEER_ID)
    return sorted(set(a) & set(b), key=lambda s: (len(s), s))


def undistort(src_png, dst_png, K, D):
    im = cv2.imread(src_png)
    if im is None:
        raise IOError("Could not read %s" % src_png)
    out = cv2.fisheye.undistortImage(im, K, D, Knew=K)
    cv2.imwrite(dst_png, out)


def load_frames(cav_id, idx):
    p = CAM_PARAMS[cav_id]
    folder_name = discover_trajectories(cav_id).get(idx)
    if folder_name is None:
        raise FileNotFoundError(
            "No converted_*_%s folder found for cav %s under %s" % (idx, cav_id, p["root"]))
    scen_dir = os.path.join(p["root"], folder_name)
    frames = []
    for fid in sorted(os.listdir(scen_dir)):
        fdir = os.path.join(scen_dir, fid)
        if not os.path.isdir(fdir) or not fid.isdigit():
            continue
        ts_path = os.path.join(fdir, "timestamp.json")
        pose_path = os.path.join(fdir, "ego_vicon_pose.json")
        if not (os.path.exists(ts_path) and os.path.exists(pose_path)):
            continue
        ts = json.load(open(ts_path))
        pose = json.load(open(pose_path))
        if not pose.get("valid", 0.0) == 1.0:
            continue
        frames.append({
            "fid": fid, "dir": fdir,
            "t": ts["reference_time_sec"],
            "x": pose["x"], "y": pose["y"], "z": pose["z"], "yaw_rad": pose["yaw"],
        })
    return frames


def pair_frames(frames_a, frames_b):
    pairs = []
    for fa in frames_a:
        fb = min(frames_b, key=lambda r: abs(r["t"] - fa["t"]))
        if abs(fb["t"] - fa["t"]) <= PAIR_TOLERANCE_SEC:
            pairs.append((fa, fb))
    return pairs


def build_agent_yaml(me, me_id, out_frame_dir, out_frame_num):
    p_me = CAM_PARAMS[me_id]
    me_yaw_deg = math.degrees(me["yaw_rad"])
    pose = [me["x"] * SCALE, me["y"] * SCALE, 0.0, 0.0, me_yaw_deg, 0.0]

    c, s = math.cos(math.radians(me_yaw_deg)), math.sin(math.radians(me_yaw_deg))
    mx, my = FRONT_MOUNT_XYZ[0], FRONT_MOUNT_XYZ[1]
    cam_cords = [me["x"] * SCALE + (c * mx - s * my) * SCALE,
                 me["y"] * SCALE + (s * mx + c * my) * SCALE,
                 FRONT_MOUNT_XYZ[2], 0.0, me_yaw_deg, 0.0]

    params = {
        "lidar_pose": pose, "lidar_pose_clean": pose,
        "true_ego_pos": pose, "ego_speed": 0.0,
        # Always empty: no ground truth for real inference. Matches this
        # project's established "negative frame" convention (see
        # visualize_boxes.py) rather than inventing a new "no GT" marker.
        "vehicles": {},
        "qcar_provenance": {
            "car": p_me["car"], "cav_id": me_id,
            "pose_source": "Vicon, REAL",
            "world_scale": "x10 horizontal only; z/height/K untouched",
            "note": "Inference-only frame -- no ground truth computed or available.",
        },
        "camera0": {
            "cords": cam_cords, "extrinsic": np.eye(4).tolist(),
            "intrinsic": p_me["K"].tolist(),
        },
    }
    with open(os.path.join(out_frame_dir, "%s.yaml" % out_frame_num), "w") as f:
        yaml.safe_dump(params, f, default_flow_style=None, sort_keys=False)


def main():
    if os.path.isdir(DST):
        shutil.rmtree(DST)

    indices = common_indices()
    if not indices:
        print("No trajectory index found on BOTH cars under %s -- nothing to build. "
              "Each car needs a converted_*_<NN> folder in its dataset/ dir "
              "sharing the same trailing index." % ONBOARD_ROOT)
        return

    manifest = {"built_for": "inference only, no ground truth", "pairs": {}}
    for idx in indices:
        frames_a = load_frames(EGO_ID, idx)
        frames_b = load_frames(PEER_ID, idx)
        if not frames_a or not frames_b:
            print("skip pair %s: missing frames (a=%d b=%d)" % (idx, len(frames_a), len(frames_b)))
            continue
        pairs = pair_frames(frames_a, frames_b)
        scen_name = "qcar_inf_%s" % idx
        n_written = 0
        for k, (fa, fb) in enumerate(pairs):
            out_num = "%06d" % k
            for me_id, me_frame in ((EGO_ID, fa), (PEER_ID, fb)):
                out_dir = os.path.join(DST, scen_name, me_id)
                os.makedirs(out_dir, exist_ok=True)
                src_png = os.path.join(me_frame["dir"], "front.png")
                dst_png = os.path.join(out_dir, "%s_camera0.png" % out_num)
                p = CAM_PARAMS[me_id]
                undistort(src_png, dst_png, p["K"], p["D"])
                dst_vis = os.path.join(out_dir, "%s_bev_visibility.png" % out_num)
                try:
                    os.link(p["vis_mask"], dst_vis)
                except OSError:
                    shutil.copy2(p["vis_mask"], dst_vis)
                build_agent_yaml(me_frame, me_id, out_dir, out_num)
            n_written += 1
        manifest["pairs"][idx] = {"frames_a": len(frames_a), "frames_b": len(frames_b),
                                   "paired": n_written}
        print("pair %s: %d/%d frames paired (tolerance %.0fms)" % (
            idx, n_written, min(len(frames_a), len(frames_b)), PAIR_TOLERANCE_SEC * 1000))

    json.dump(manifest, open(os.path.join(DST, "manifest.json"), "w"), indent=2)
    print("\nwrote", DST)
    print("Point a hypes yaml's validate_dir (or test_dir) at this folder to run "
          "qcar/eval.py or qcar/visualize.py for pure "
          "inference -- recall/AP numbers will read as 0 (there is no GT to score "
          "against), which is expected, not a bug.")


if __name__ == "__main__":
    main()
