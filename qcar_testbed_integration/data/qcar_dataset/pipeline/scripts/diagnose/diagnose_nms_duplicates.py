"""Root-cause check for PENDIENTE item #2: duplicate boxes on the same
target in the cooperative case (frame idx=28, Node 9: scores 0.57 and 0.25,
not merged because nms_thresh=0.15 was insufficient -- see ERRORES #12 and
RESULTADOS). Two questions this script answers with real numbers instead of
the single hand-inspected frame:

1. Would raising nms_thresh actually merge most Node-9 duplicate clusters?
2. What would it cost elsewhere? The two QCars themselves get as close as
   0.65m in this dataset (see pipeline/CONVERSION_NOTES.md) --
   raising nms_thresh globally risks merging two genuinely distinct nearby
   detections into one, silently costing real recall.

Caches one forward pass per frame, then re-runs ONLY the postprocessing
(decode already done by the model; nms_thresh is read from
post_processor.params at call time, so no retraining/recompute needed) at
each candidate threshold.

    python diagnose_nms_duplicates.py --model_dir opencood/logs/<coop_run>
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
# from the HEAL repo root (see the root README.md, Installation) -- no path hack needed.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "cooperative"))

import qcar.patches.patch_1cam_loader  # noqa: E402,F401
import qcar.patches.patch_real_extrinsic  # noqa: E402,F401
from opencood.data_utils.datasets import build_dataset  # noqa: E402
from opencood.tools import train_utils  # noqa: E402
from opencood.utils import eval_utils  # noqa: E402

from build_coop_train_val_dataset import STATIC_TARGETS, SCALE, apply_marker_offset, truly_visible_to_agent  # noqa: E402
from diagnose_coop_breakdown import expected_relative_center, DIST_THRESH_X10  # noqa: E402

TARGET_ID = "3"  # Node 9, the target with the documented duplicate-box case
NMS_CANDIDATES = [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
IOU_THRESH = 0.2  # matches eval_qcar.py's recall@IoU convention


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

    cached = []  # (batch_data, output_dict, ego_pose, peer_sees_node9, expected_xy)
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
            cx, cy, _ = expected_relative_center(target_world, ego)

            batch_data = train_utils.to_device(batch_data, device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                output_dict = model(batch_data["ego"])
            # move to CPU before caching -- this GPU is 3.68GiB total (see ERRORES #9)
            # and can't hold 237 frames' worth of activations; post_process's own NMS
            # step already does .cpu().numpy() internally, so running it on CPU-cached
            # tensors for the threshold sweep is safe, just a bit slower.
            batch_data = train_utils.to_device(batch_data, "cpu")
            output_dict = train_utils.to_device(output_dict, "cpu")
            cached.append((batch_data, output_dict, peer_sees, (cx, cy)))

    print("cached %d frames\n" % len(cached))
    print("%8s %14s %16s %14s %14s" % ("nms_th", "node9_multi", "node9_frames_gte2",
                                        "global_tp", "global_fp"))
    for t in NMS_CANDIDATES:
        ds.post_processor.params["nms_thresh"] = t
        result_stat = {IOU_THRESH: {"tp": [], "fp": [], "score": [], "gt": 0}}
        multi_counts = []
        for batch_data, output_dict, peer_sees, (cx, cy) in cached:
            pred_box_tensor, pred_score, gt_box_tensor = ds.post_process(batch_data, {"ego": output_dict})
            eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, IOU_THRESH)

            if not peer_sees or pred_box_tensor is None or pred_box_tensor.shape[0] == 0:
                continue
            centers = pred_box_tensor.cpu().numpy().mean(axis=1)
            d = np.linalg.norm(centers[:, :2] - np.array([cx, cy]), axis=1)
            n_near = int((d < DIST_THRESH_X10).sum())
            if n_near > 0:
                multi_counts.append(n_near)

        stat = result_stat[IOU_THRESH]
        tp, fp = sum(stat["tp"]), sum(stat["fp"])
        n_ge2 = sum(1 for n in multi_counts if n >= 2)
        mean_multi = (sum(multi_counts) / len(multi_counts)) if multi_counts else float("nan")
        print("%8.2f %14.2f %16d %14d %14d" % (t, mean_multi, n_ge2, tp, fp))

    print("\nnode9_multi = mean number of surviving boxes within %.1fm of Node9's expected\n"
          "  center, over frames where the peer actually sees it (1.0 = clean, no dup)\n"
          "node9_frames_gte2 = how many of those frames still have >=2 competing boxes\n"
          "global_tp/fp = whole-dataset recall/precision impact at that nms_thresh\n"
          "  (if fp drops with thresh but tp ALSO drops noticeably, raising nms_thresh is\n"
          "  merging real distinct detections too, not just Node9 duplicates)" % DIST_THRESH_X10)


if __name__ == "__main__":
    main()
