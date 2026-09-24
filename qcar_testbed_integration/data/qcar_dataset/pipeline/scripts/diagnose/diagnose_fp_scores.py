"""Root-cause check for the 37% false-positive rate on Node 9 (target "3",
wall-blocked, cooperation-only) documented in the 2026-09-14 PENDIENTE note.

Hypothesis under test: those false positives carry systematically lower
confidence scores than the true positives on the SAME target, so raising
`postprocess.target_args.score_threshold` (currently 0.2, see
resolved_hypes.json) would clean most of them up without costing much real
recall.

For every validate frame, finds the nearest surviving prediction (already
decode+NMS+score_threshold-filtered by VoxelPostprocessor, same as training
labels) to Node 9's expected relative position, using the SAME distance
convention as diagnose_coop_breakdown.py (DIST_THRESH_X10=3.0, i.e. ~0.3m).
Buckets that nearest-prediction's score by whether the peer (.158) actually
sees Node 9 in that frame:
  - peer sees it  -> a matched nearby prediction is a real true positive
  - peer blind    -> a matched nearby prediction is the false positive being
                      investigated (ego itself is wall-blocked from Node 9
                      by real wall geometry, see wall_geometry.py)
Then reports the two score distributions and a threshold sweep, so the
decision to raise score_threshold is based on the actual numbers instead of
the single frame (idx=28) the original note inspected by hand.

    python diagnose_fp_scores.py --model_dir opencood/logs/<coop_run>
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

TARGET_ID = "3"       # Node 9, the wall-cooperation-only case this note is about
TARGET_NODE = "Node9"


def summarize(name, scores):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        print("%-24s n=0" % name)
        return
    qs = np.percentile(scores, [0, 25, 50, 75, 100])
    print("%-24s n=%3d  mean=%.3f  min=%.3f  p25=%.3f  median=%.3f  p75=%.3f  max=%.3f"
          % (name, scores.size, scores.mean(), qs[0], qs[1], qs[2], qs[3], qs[4]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--no_amp", action="store_true")
    opt = ap.parse_args()

    hypes = json.load(open(os.path.join(opt.model_dir, "resolved_hypes.json")))
    current_threshold = hypes["postprocess"]["target_args"]["score_threshold"]
    ds = build_dataset(dict(hypes, validate_dir=hypes["validate_dir"]), visualize=False, train=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=ds.collate_batch_test)

    model = train_utils.create_model(hypes)
    _, model = train_utils.load_saved_model(opt.model_dir, model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    use_amp = bool(torch.cuda.is_available() and not opt.no_amp)

    tp_scores, fp_scores = [], []   # nearest-prediction score, bucketed by peer visibility
    n_visible, n_blind = 0, 0
    n_visible_missed, n_blind_clean = 0, 0
    fp_peer_dist, clean_peer_dist = [], []   # peer-to-target real distance (m), blind frames only
    fp_with_real_gt, fp_without_real_gt = 0, 0   # is it actually in HEAL's own gt_box_tensor?

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

            batch_data = train_utils.to_device(batch_data, device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                output_dict = model(batch_data["ego"])
            pred_box_tensor, pred_score, gt_box_tensor = ds.post_process(batch_data, {"ego": output_dict})

            if pred_box_tensor is None or pred_box_tensor.shape[0] == 0:
                pred_centers = np.zeros((0, 3))
                pred_scores_np = np.zeros((0,))
            else:
                pred_centers = pred_box_tensor.cpu().numpy().mean(axis=1)
                pred_scores_np = pred_score.cpu().numpy()

            cx, cy, _ = expected_relative_center(target_world, ego)
            nearest_score = None
            if pred_centers.shape[0] > 0:
                d = np.linalg.norm(pred_centers[:, :2] - np.array([cx, cy]), axis=1)
                j = int(d.argmin())
                if d[j] < DIST_THRESH_X10:
                    nearest_score = float(pred_scores_np[j])

            if peer_sees:
                n_visible += 1
                if nearest_score is not None:
                    tp_scores.append(nearest_score)
                else:
                    n_visible_missed += 1
            else:
                n_blind += 1
                peer_dist = math.hypot(peer["x"] - target_world["x"], peer["y"] - target_world["y"])
                if nearest_score is not None:
                    fp_scores.append(nearest_score)
                    fp_peer_dist.append(peer_dist)
                    has_real_gt = False
                    if gt_box_tensor is not None and gt_box_tensor.shape[0] > 0:
                        gt_centers = gt_box_tensor.cpu().numpy().mean(axis=1)
                        gd = np.linalg.norm(gt_centers[:, :2] - np.array([cx, cy]), axis=1)
                        has_real_gt = bool(gd.min() < DIST_THRESH_X10)
                    if has_real_gt:
                        fp_with_real_gt += 1
                    else:
                        fp_without_real_gt += 1
                else:
                    n_blind_clean += 1
                    clean_peer_dist.append(peer_dist)

    print("Node 9 (target %r), current score_threshold=%.2f, nms_thresh=%.2f, dist_thresh_x10=%.1f\n"
          % (TARGET_ID, current_threshold, hypes["postprocess"]["nms_thresh"], DIST_THRESH_X10))
    print("frames peer sees Node9:  %d  (matched=%d, missed=%d)"
          % (n_visible, len(tp_scores), n_visible_missed))
    print("frames peer blind:       %d  (false-positive matched=%d, clean=%d)\n"
          % (n_blind, len(fp_scores), n_blind_clean))

    summarize("true-positive scores", tp_scores)
    summarize("false-positive scores", fp_scores)

    print("\nthird check: of the matched-FP frames, how many does HEAL's OWN real\n"
          "gt_box_tensor (the actual training/eval ground truth, via the coarse\n"
          "box_is_visible() BEV mask -- NOT the strict truly_visible_to_agent used\n"
          "above) already count as a legitimate box near Node9? If most of them do,\n"
          "these are not eval_qcar.py false positives at all -- the model is matching\n"
          "the GT it was actually trained/evaluated against, and the mismatch is\n"
          "between two DIFFERENT visibility oracles, not a model or training defect.")
    print("fp frames WITH a real HEAL gt box nearby:    %d" % fp_with_real_gt)
    print("fp frames WITHOUT a real HEAL gt box nearby: %d" % fp_without_real_gt)

    print("\nsecond hypothesis: are false positives concentrated where the peer is JUST\n"
          "outside its visibility range (near-miss / smooth decision boundary) rather\n"
          "than uniformly spread over all peer-blind frames (pure noise/overfit)?")
    summarize("peer dist, FP frames (m)", fp_peer_dist)
    summarize("peer dist, clean frames (m)", clean_peer_dist)
    BOUNDARY_M = 1.0  # ~3x the FP median distance; frames within this of the
                       # cutoff are "arguably still visible", farther ones
                       # would be unexplained hallucination
    near = sum(1 for d in fp_peer_dist if d < BOUNDARY_M)
    far = sum(1 for d in fp_peer_dist if d >= BOUNDARY_M)
    print("of the %d matched-FP frames: %d are within %.1fm of the visibility cutoff "
          "(boundary-explainable), %d are genuinely far (unexplained)"
          % (len(fp_peer_dist), near, BOUNDARY_M, far))

    print("\nthreshold sweep (recall on peer-visible frames vs FP rate on peer-blind frames):")
    print("%8s %10s %10s" % ("thresh", "tp_kept", "fp_kept"))
    for t in np.arange(0.2, 1.0, 0.05):
        tp_kept = sum(1 for s in tp_scores if s >= t)
        fp_kept = sum(1 for s in fp_scores if s >= t)
        print("%8.2f %10d %10d" % (t, tp_kept, fp_kept))


if __name__ == "__main__":
    main()
