"""Cross-check: of the 35 Node9 false-positive frames (strict oracle,
margin=0), are the ones NOT explained by HEAL's real GT (diagnose_fp_scores.py's
check) the SAME frames as the ones NOT explained by a widened FOV margin
(diagnose_visibility_margin.py's check)? Those two scripts ran independent
filters over the same 35 frames but never compared them frame-by-frame.

This script runs a single forward pass per frame and, for each of the 35
matched-FP frames, records:
  A) has_real_gt   -- does HEAL's own gt_box_tensor (box_is_visible mask)
                       already have a box near Node9 here?
  B) margin_visible -- does widening truly_visible_to_agent's FOV check by
                       50% (MARGIN_FRAC) reclassify this frame as "visible"?
Then reports the 2x2 breakdown and the exact frame indices in the
"neither A nor B" bucket -- the genuinely unexplained residual.

    python diagnose_fp_overlap.py --model_dir opencood/logs/<coop_run>
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
    STATIC_TARGETS, SCALE, apply_marker_offset, truly_visible_to_agent,
)
from diagnose_coop_breakdown import expected_relative_center, DIST_THRESH_X10  # noqa: E402
from diagnose_visibility_margin import visible_with_margin  # noqa: E402

TARGET_ID = "3"
MARGIN_FRAC = 0.50


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

    fp_frames = []  # (idx, has_real_gt, margin_visible, peer_dist)
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

            peer_sees = truly_visible_to_agent(
                target_world["x"], target_world["y"], target_world["z"],
                math.degrees(target_world["yaw_rad"]), peer, "2")
            if peer_sees:
                continue  # only care about the "peer blind" bucket (95 frames)

            batch_data = train_utils.to_device(batch_data, device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                output_dict = model(batch_data["ego"])
            pred_box_tensor, pred_score, gt_box_tensor = ds.post_process(batch_data, {"ego": output_dict})

            cx, cy, _ = expected_relative_center(target_world, ego)
            nearest_score = None
            if pred_box_tensor is not None and pred_box_tensor.shape[0] > 0:
                pred_centers = pred_box_tensor.cpu().numpy().mean(axis=1)
                d = np.linalg.norm(pred_centers[:, :2] - np.array([cx, cy]), axis=1)
                j = int(d.argmin())
                if d[j] < DIST_THRESH_X10:
                    nearest_score = float(pred_score.cpu().numpy()[j])

            if nearest_score is None:
                continue  # clean frame, not one of the 35

            has_real_gt = False
            if gt_box_tensor is not None and gt_box_tensor.shape[0] > 0:
                gt_centers = gt_box_tensor.cpu().numpy().mean(axis=1)
                gd = np.linalg.norm(gt_centers[:, :2] - np.array([cx, cy]), axis=1)
                has_real_gt = bool(gd.min() < DIST_THRESH_X10)

            margin_visible = visible_with_margin(
                target_world["x"], target_world["y"], target_world["z"],
                math.degrees(target_world["yaw_rad"]), peer, "2", MARGIN_FRAC)

            peer_dist = math.hypot(peer["x"] - target_world["x"], peer["y"] - target_world["y"])
            fp_frames.append((idx, has_real_gt, margin_visible, peer_dist))

    print("%d matched-FP frames (expect 35)\n" % len(fp_frames))

    both = [f for f in fp_frames if f[1] and f[2]]
    only_a = [f for f in fp_frames if f[1] and not f[2]]
    only_b = [f for f in fp_frames if not f[1] and f[2]]
    neither = [f for f in fp_frames if not f[1] and not f[2]]

    print("A = has real HEAL gt box nearby (box_is_visible)")
    print("B = becomes 'visible' under a %.0f%% wider FOV margin\n" % (MARGIN_FRAC * 100))
    print("both A and B (explained twice over):        %d" % len(both))
    print("only A (real GT, but margin doesn't cover):  %d" % len(only_a))
    print("only B (margin covers, but no real GT):      %d" % len(only_b))
    print("neither A nor B (truly unexplained):         %d" % len(neither))
    print("\nframe indices, neither A nor B:", [f[0] for f in neither])
    print("their peer distances (m):", ["%.2f" % f[3] for f in neither])
    print("\nframe indices, only A:", [f[0] for f in only_a])
    print("frame indices, only B:", [f[0] for f in only_b])


if __name__ == "__main__":
    main()
