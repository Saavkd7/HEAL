"""Correct the pretrain manifest's positive/negative flag: Vicon-valid is not
camera-visible. Reuses the already-verified qcar-52775 visibility mask
(same physical camera, confirmed by Kevin) and the project's own
box_is_visible() -- no new geometry derived, no new mask built.

relative_vicon_pose.json already gives the target's position in the ego's
own forward/lateral frame (x_forward_m, y_lateral_m), so no extra transform
is needed -- only the x10 horizontal scale the rest of the project applies
before touching box_is_visible(), which was built in that scaled space.

Writes pretrain_labels.json: same per-scenario frame lists as the manifest,
but `has_target` replaced by `camera_visible` (mask-checked) plus the box
(x,y,z,l,w,h,yaw) in scaled model space for frames that pass.

2026-09-14 correction: the 256x256 BEV visibility mask is a coarse cell-grid
approximation of the camera's FOV. An audit (audit_box_projection.py)
projecting every "visible" box with the real K/extrinsic found 26%
(1218/4680) actually land outside the 640x480 image or behind the camera --
real label noise, not a rare edge case. A frame now needs BOTH: inside the
mask's wedge AND the box's projected centroid inside the real image bounds.
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

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
    "--manifest-json", default=_path_from_conf("manifest_json"),
    help="pretrain_manifest.json path (default: conf.json's manifest_json)")
_parser.add_argument(
    "--labels-json", default=_path_from_conf("labels_json"),
    help="pretrain_labels.json path (default: conf.json's labels_json)")
_parser.add_argument(
    "--ego-vis-mask", default=_path_from_conf("ego_vis_mask"),
    help="Ego car's BEV visibility mask png (default: conf.json's ego_vis_mask)")
if __name__ == "__main__" and any(a in ("-h", "--help") for a in sys.argv[1:]):
    _parser.print_help()
    raise SystemExit(0)
_args, _ = _parser.parse_known_args()
_missing = [k for k in ("src_root", "calib_root", "manifest_json", "labels_json", "ego_vis_mask")
            if getattr(_args, k) is None]
if _missing:
    raise SystemExit("No value for %s: set it in %s or pass the flag."
                     % (", ".join("--" + k.replace("_", "-") for k in _missing), _conf_path))

SRC_ROOT = _args.src_root
MANIFEST_PATH = _args.manifest_json
OUT_PATH = _args.labels_json
MASK_PATH = _args.ego_vis_mask
CALIB = _args.calib_root

SCALE = 10.0
QCAR_L, QCAR_W, QCAR_H = 0.425, 0.192, 0.190
FRONT_MOUNT_XYZ = (0.1930, 0.0, 0.0953)
IMG_W, IMG_H = 640, 480

# opencood is importable directly once `python setup.py develop` has been run
# from the HEAL repo root (see the root README.md, Installation) -- no path hack needed.
sys.path.insert(0, os.path.join(PIPELINE_ROOT, "scripts", "visualize"))
from opencood.utils.box_utils import box_is_visible  # noqa: E402
from visualize_boxes import box_corners_world, project  # noqa: E402

_calib = np.load(os.path.join(CALIB, "qcar52775/latest_front",
                              "qcar52775_front_intrinsics_verified.npz"))
FRONT_K = _calib["front_K"].astype(np.float64)
CAM_CORDS = [FRONT_MOUNT_XYZ[0] * SCALE, FRONT_MOUNT_XYZ[1] * SCALE,
             FRONT_MOUNT_XYZ[2], 0.0, 0.0, 0.0]


def truly_in_frame(x, y, z, l, w, h, yaw_deg):
    corners = box_corners_world([x, y, z], [0.0, 0.0, h / 2.0],
                                [l / 2.0, w / 2.0, h / 2.0], yaw_deg)
    uv, valid, depth = project(corners, CAM_CORDS, FRONT_K)
    if valid.sum() < 4:
        return False
    center = uv[valid].mean(axis=0)
    return 0 <= center[0] < IMG_W and 0 <= center[1] < IMG_H


def main() -> None:
    manifest = json.load(open(MANIFEST_PATH))
    vis_map = np.array(Image.open(MASK_PATH))
    print(f"visibility mask: {vis_map.shape}, {int((vis_map > 0).sum())} cells lit")

    out = {"source": SRC_ROOT, "mask_used": MASK_PATH, "scale": SCALE,
           "scenarios": {}}
    tot_frames = tot_visible = tot_vicon_valid_not_visible = 0

    for si, (scen_name, scen) in enumerate(manifest["scenarios"].items(), 1):
        scen_dir = os.path.join(SRC_ROOT, scen_name)
        kept_ids = scen["kept_frame_ids"]
        kept_has_target = scen["kept_has_target"]
        labels = []

        for fid, had_target in zip(kept_ids, kept_has_target):
            tot_frames += 1
            if not had_target:
                labels.append({"frame_id": fid, "camera_visible": False})
                continue

            rel_path = os.path.join(scen_dir, fid, "relative_vicon_pose.json")
            try:
                rel = json.loads(open(rel_path).read())
            except Exception:
                labels.append({"frame_id": fid, "camera_visible": False})
                continue
            if not rel.get("both_valid", False):
                labels.append({"frame_id": fid, "camera_visible": False})
                continue

            x = rel["x_forward_m"] * SCALE
            y = rel["y_lateral_m"] * SCALE
            yaw_deg = rel.get("relative_yaw_deg", 0.0)
            bbx = np.array([[x, y, 0.0, 0.0, 0.0, 0.0, np.deg2rad(yaw_deg)]])
            mask_visible = bool(box_is_visible(bbx, vis_map))
            visible = mask_visible and truly_in_frame(
                x, y, 0.1226, QCAR_L * SCALE, QCAR_W * SCALE, QCAR_H, yaw_deg)

            if visible:
                tot_visible += 1
                labels.append({
                    "frame_id": fid, "camera_visible": True,
                    "box_xyzlwh_yaw_deg": [
                        x, y, 0.1226 * 1.0,  # z: unscaled, real QCar body height off ground
                        QCAR_L * SCALE, QCAR_W * SCALE, QCAR_H,
                        yaw_deg],
                })
            else:
                tot_vicon_valid_not_visible += 1
                labels.append({"frame_id": fid, "camera_visible": False})

        out["scenarios"][scen_name] = {"labels": labels}
        if si % 20 == 0 or si == len(manifest["scenarios"]):
            print(f"[{si}/{len(manifest['scenarios'])}] scenarios, "
                  f"{tot_visible} visible so far", flush=True)

    out["totals"] = {
        "frames_checked": tot_frames,
        "camera_visible_positive": tot_visible,
        "vicon_valid_but_not_visible": tot_vicon_valid_not_visible,
        "negative_no_target": tot_frames - tot_visible - tot_vicon_valid_not_visible,
    }
    json.dump(out, open(OUT_PATH, "w"), indent=2)
    print(f"\nWrote {OUT_PATH}")
    print(json.dumps(out["totals"], indent=2))


if __name__ == "__main__":
    main()
