"""Follow-up to diagnose_fp_scores.py's hypothesis 2 (05 SEGUIMIENTO,
2026-09-17): if the "false positives" on Node 9 are mostly boundary
near-misses (target genuinely still partly visible, oracle says no because
its check is a hard binary pixel-bounds cutoff with zero tolerance), then
WIDENING that cutoff -- a bigger FOV cone / a pixel-bounds margin -- should
reclassify most of those frames as "peer sees it" and shrink the count of
truly-unexplained false positives.

This is a DIAGNOSTIC-ONLY variant of build_coop_train_val_dataset.py's
truly_visible_to_agent -- it does NOT change the real oracle used to build
scene_gt/training labels. It only asks: if we relax the visibility check by
a margin, does that change what we'd call "false positive" on THIS already-
trained checkpoint's predictions?

Single forward pass per frame (cached), then the margin sweep only redoes
the geometry check (cheap), never re-runs the model.

    python diagnose_visibility_margin.py --model_dir opencood/logs/<coop_run>
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

# opencood is importable directly once `python setup.py develop` has been run
# from the HEAL repo root (see its own CLAUDE.md) -- no path hack needed.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "cooperative"))

import qcar.patches.patch_1cam_loader  # noqa: E402,F401
import qcar.patches.patch_real_extrinsic  # noqa: E402,F401
from opencood.data_utils.datasets import build_dataset  # noqa: E402
from opencood.tools import train_utils  # noqa: E402

from build_coop_train_val_dataset import (  # noqa: E402
    STATIC_TARGETS, SCALE, apply_marker_offset,
    world_box_corners, project_world_to_camera, CAM_PARAMS,
    QCAR_L, QCAR_W, QCAR_H, IMG_W, IMG_H,
)
from diagnose_coop_breakdown import expected_relative_center, DIST_THRESH_X10  # noqa: E402

TARGET_ID = "3"  # Node 9
MARGIN_FRACS = [0.00, 0.10, 0.20, 0.30, 0.50]  # fraction of image W/H added on EACH side


def visible_with_margin(target_x, target_y, target_z, target_yaw_deg, agent_pose, cav_id, margin_frac):
    """Same check as build_coop_train_val_dataset.truly_visible_to_agent, but the
    image-bounds test is widened by margin_frac * (W, H) on every side --
    a crude stand-in for 'the real FOV cone is a bit wider than modeled' or
    'calibration/pose has some tolerance', not a physically fit number."""
    corners = world_box_corners(target_x, target_y, target_z, target_yaw_deg, QCAR_L, QCAR_W, QCAR_H)
    K = CAM_PARAMS[cav_id]["K"]
    uv, depth = project_world_to_camera(corners, agent_pose["x"], agent_pose["y"], agent_pose["z"],
                                         agent_pose["yaw_rad"], cav_id, K)
    valid = depth > 0.05
    if valid.sum() < 4:
        return False
    center = uv[valid].mean(axis=0)
    mx, my = margin_frac * IMG_W, margin_frac * IMG_H
    return bool(-mx <= center[0] < IMG_W + mx and -my <= center[1] < IMG_H + my)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--no_amp", action="store_true")
    opt = ap.parse_args()

    hypes = json.load(open(os.path.join(opt.model_dir, "resolved_hypes.json")))
    ds = build_dataset(dict(hypes, validate_dir=hypes["validate_dir"]), visualize=False, train=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=ds.collate_batch_test)

    model = train_utils.create_model(hypes)
    _, model = train_utils.load_saved_model(opt.model_dir, model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    use_amp = bool(torch.cuda.is_available() and not opt.no_amp)

    frames = []  # (peer_pose, target_world, nearest_score_or_None)
    with torch.inference_mode():
        for idx, batch_data in enumerate(loader):
            if batch_data is None:
                continue
            base = ds.retrieve_base_data(idx)
            if "1" not in base or "2" not in base:
                continue
            p1 = base["1"]["params"]["lidar_pose"]
            p2 = base["2"]["params"]["lidar_pose"]
            ego = {"x": p1[0] / SCALE, "y": p1[1] / SCALE, "z": 0.0, "yaw_rad": math.radians(p1[4])}
            peer = {"x": p2[0] / SCALE, "y": p2[1] / SCALE, "z": 0.0, "yaw_rad": math.radians(p2[4])}
            target_world = apply_marker_offset(dict(STATIC_TARGETS[TARGET_ID]))

            batch_data = train_utils.to_device(batch_data, device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                output_dict = model(batch_data["ego"])
            pred_box_tensor, pred_score, _ = ds.post_process(batch_data, {"ego": output_dict})

            nearest_score = None
            if pred_box_tensor is not None and pred_box_tensor.shape[0] > 0:
                pred_centers = pred_box_tensor.cpu().numpy().mean(axis=1)
                pred_scores_np = pred_score.cpu().numpy()
                cx, cy, _ = expected_relative_center(target_world, ego)
                d = np.linalg.norm(pred_centers[:, :2] - np.array([cx, cy]), axis=1)
                j = int(d.argmin())
                if d[j] < DIST_THRESH_X10:
                    nearest_score = float(pred_scores_np[j])

            frames.append((peer, target_world, nearest_score))

    print("%d frames cached\n" % len(frames))
    print("%10s %10s %10s %14s %20s" % ("margin", "n_visible", "n_blind", "fp(blind+det)",
                                         "newly-visible-w-det"))
    base_blind_fp = None
    for m in MARGIN_FRACS:
        n_visible = n_blind = fp = 0
        reclassified_with_detection = 0
        for peer, target_world, nearest_score in frames:
            sees = visible_with_margin(
                target_world["x"], target_world["y"], target_world["z"],
                math.degrees(target_world["yaw_rad"]), peer, "2", m)
            if sees:
                n_visible += 1
            else:
                n_blind += 1
                if nearest_score is not None:
                    fp += 1
        if m == 0.0:
            base_blind_fp = fp
        print("%10.2f %10d %10d %14d %20s" % (m, n_visible, n_blind, fp, "-"))

    print("\nbaseline (margin=0, the real oracle used for training): %d blind frames with a "
          "matched detection (the '35 falsos positivos' number)." % base_blind_fp)
    print("if a wider FOV margin genuinely explains most of them, fp should drop sharply as\n"
          "margin grows AND n_blind should shrink by roughly the same amount (frames moving\n"
          "from 'blind' to 'visible', not vanishing) -- watch both columns together, not fp alone.")


if __name__ == "__main__":
    main()
