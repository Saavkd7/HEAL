"""Per-target-category diagnostic for the cooperation model: split recall
into (a) the dynamic peer -- id "2", usually directly visible to ego's own
camera -- vs (b) target "3" (Node 9, wall-blocked for ego=car1 -- ONLY
detectable via cooperation with car2) vs (c) target "4" (Node 11, out of
ego's own FOV early on, in range later -- ordinary geometric visibility).
This is the honest answer to "did we actually learn to see through the
wall, or does the model just do well on the easy dynamic-car case and
fail on the hard cooperation-only case" -- a single blended recall number
cannot tell the two apart.

    python qcar_dataset/pipeline/shared/evidence_through_wall/diagnose_coop_breakdown.py \
        --model_dir opencood/logs/<coop_run> [--no_amp]
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/mnt/mainvolume/Backup/Projects/HEAL")
sys.path.insert(0, "/mnt/mainvolume/Backup/Projects/HEAL/qcar_dataset/Inference")

import qcar.patches.patch_1cam_loader  # noqa: E402,F401
import qcar.patches.patch_real_extrinsic  # noqa: E402,F401
from opencood.data_utils.datasets import build_dataset  # noqa: E402
from opencood.tools import train_utils  # noqa: E402

from build_coop_train_val_dataset import (  # noqa: E402
    STATIC_TARGETS, WALL_BLOCKED, SCALE, apply_marker_offset, truly_visible_to_agent,
)

DIST_THRESH_X10 = 3.0  # ~0.3m real; a predicted box within this of the expected
                       # center counts as "found" -- matches this project's
                       # established close-target separation distance.


def expected_relative_center(target_world, ego_pose_real):
    """target_world/ego_pose_real: dict with x,y,z,yaw_rad (real metres).
    Returns (x10, y10, z_real) in ego's lidar frame, matching post_process's
    output convention."""
    ex, ey, ez, eyaw = ego_pose_real["x"], ego_pose_real["y"], ego_pose_real["z"], ego_pose_real["yaw_rad"]
    c, s = math.cos(eyaw), math.sin(eyaw)
    dx, dy = target_world["x"] - ex, target_world["y"] - ey
    rel_x = c * dx + s * dy   # R_ego^T @ (world - ego), row-vector form
    rel_y = -s * dx + c * dy
    return rel_x * SCALE, rel_y * SCALE, target_world["z"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--no_amp", action="store_true")
    opt = ap.parse_args()

    hypes = json.load(open(os.path.join(opt.model_dir, "resolved_hypes.json")))
    h2 = dict(hypes)
    h2["validate_dir"] = hypes["validate_dir"]
    ds = build_dataset(h2, visualize=False, train=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=ds.collate_batch_test)

    model = train_utils.create_model(hypes)
    _, model = train_utils.load_saved_model(opt.model_dir, model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    use_amp = bool(torch.cuda.is_available() and not opt.no_amp)

    stats = {
        "2_dynamic_peer": {"expected": 0, "found": 0},
        "3_wall_cooperation_only": {"expected": 0, "found": 0},
        "4_own_camera_fov": {"expected": 0, "found": 0},
    }

    with torch.inference_mode():
        for idx, batch_data in enumerate(loader):
            if batch_data is None:
                continue
            base = ds.retrieve_base_data(idx)
            if "1" not in base or "2" not in base:
                continue
            p1 = base["1"]["params"]["lidar_pose"]
            p2 = base["2"]["params"]["lidar_pose"]
            ego = {"x": p1[0] / SCALE, "y": p1[1] / SCALE, "z": 0.0,
                   "yaw_rad": math.radians(p1[4])}
            peer = {"x": p2[0] / SCALE, "y": p2[1] / SCALE, "z": 0.0,
                    "yaw_rad": math.radians(p2[4])}

            targets_world = {vid: apply_marker_offset(
                {**STATIC_TARGETS[vid], "yaw_rad": STATIC_TARGETS[vid]["yaw_rad"]})
                for vid in ("3", "4")}
            peer_world = apply_marker_offset(peer)

            vis3_to_1 = False if ("1", "3") in WALL_BLOCKED else truly_visible_to_agent(
                targets_world["3"]["x"], targets_world["3"]["y"], targets_world["3"]["z"],
                math.degrees(targets_world["3"]["yaw_rad"]), ego, "1")
            vis3_to_2 = truly_visible_to_agent(
                targets_world["3"]["x"], targets_world["3"]["y"], targets_world["3"]["z"],
                math.degrees(targets_world["3"]["yaw_rad"]), peer, "2")
            vis4_to_1 = truly_visible_to_agent(
                targets_world["4"]["x"], targets_world["4"]["y"], targets_world["4"]["z"],
                math.degrees(targets_world["4"]["yaw_rad"]), ego, "1")
            vis4_to_2 = False if ("2", "4") in WALL_BLOCKED else truly_visible_to_agent(
                targets_world["4"]["x"], targets_world["4"]["y"], targets_world["4"]["z"],
                math.degrees(targets_world["4"]["yaw_rad"]), peer, "2")

            batch_data = train_utils.to_device(batch_data, device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                output_dict = model(batch_data["ego"])
            pred_box_tensor, pred_score, gt_box_tensor = ds.post_process(
                batch_data, {"ego": output_dict})
            if pred_box_tensor is None or pred_box_tensor.shape[0] == 0:
                pred_centers = np.zeros((0, 3))
            else:
                pred_centers = pred_box_tensor.cpu().numpy().mean(axis=1)

            def check(category, target_world_pose, visible_any):
                if not visible_any:
                    return
                stats[category]["expected"] += 1
                cx, cy, cz = expected_relative_center(target_world_pose, ego)
                if pred_centers.shape[0] == 0:
                    return
                d = np.linalg.norm(pred_centers[:, :2] - np.array([cx, cy]), axis=1)
                if d.min() < DIST_THRESH_X10:
                    stats[category]["found"] += 1

            check("2_dynamic_peer", peer_world, True)
            check("3_wall_cooperation_only", targets_world["3"], vis3_to_1 or vis3_to_2)
            check("4_own_camera_fov", targets_world["4"], vis4_to_1 or vis4_to_2)

    print("category | expected | found | recall")
    for cat, s in stats.items():
        recall = (s["found"] / s["expected"]) if s["expected"] else float("nan")
        print("%s | %d | %d | %.3f" % (cat, s["expected"], s["found"], recall))


if __name__ == "__main__":
    main()
