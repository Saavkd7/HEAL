"""Lean inference/evaluation for the QCar-real fine-tuning runs -- no open3d,
no visualisation, just decode + NMS + the recall@IoU metric this project has
used since session37 (target found within a given IoU of the true box).

Reuses HEAL's own decode+NMS path
(IntermediateheterFusionDataset.post_process -> VoxelPostprocessor.post_process,
the SAME code that generates training labels) and its own TP/FP counter
(opencood.utils.eval_utils.caluclate_tp_fp) rather than re-deriving IoU
matching from scratch -- one less place for a silent convention mismatch.

    python opencood/qcar_patches/eval_qcar.py \
        --model_dir opencood/logs/<run> --split validate

WHY THIS SCRIPT EXISTS SEPARATELY FROM opencood/tools/inference.py
-----------------------------------------------------------------------
inference.py imports open3d for LiDAR point-cloud visualisation, which is
irrelevant (and possibly unavailable) for a camera-only run, and it writes
visualisation frames by default. This script only computes numbers.

READ BEFORE TRUSTING A NUMBER FROM THIS SCRIPT
--------------------------------------------------
--split test must be run EXACTLY ONCE, after all training/hyperparameter
decisions are final. Running it repeatedly while iterating on the model
turns the held-out test set into a second validation set -- silently
invalidating the whole point of holding it out. Use --split validate while
iterating.
"""
import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader

import opencood.qcar_patches.patch_1cam_loader  # noqa: F401
import opencood.qcar_patches.patch_real_extrinsic  # noqa: F401  (real camera rotation, not CARLA's mirror)
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.utils import eval_utils


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_dir", required=True,
                    help="directory produced by train_qcar.py "
                         "(contains resolved_hypes.json + a net_epoch_bestval_at*.pth)")
    ap.add_argument("--split", choices=["train", "validate", "test"], default="validate")
    ap.add_argument("--iou_thresh", type=float, default=0.2,
                    help="matches the recall@0.2 convention used since session37")
    ap.add_argument("--no_amp", action="store_true",
                    help="disable CUDA mixed precision (enabled by default on CUDA)")
    return ap.parse_args()


def main():
    opt = parse_args()
    # resolved_hypes.json is the FULLY resolved config train_qcar.py dumped
    # next to the checkpoints -- every override (Ncams, depth_supervision,
    # assignment_path, split dirs) already applied, so this is what actually
    # trained the model, not a re-derivation that could silently drift from it.
    hypes = json.load(open(os.path.join(opt.model_dir, "resolved_hypes.json")))

    dir_key = {"train": "root_dir", "validate": "validate_dir", "test": "test_dir"}[opt.split]
    split_dir = hypes.get(dir_key)
    if not split_dir:
        raise RuntimeError(
            "%s is not configured. This development dataset intentionally has "
            "no same-trajectory test; evaluate the final model on a new, "
            "independent QCar trajectory." % dir_key
        )
    h2 = dict(hypes)
    h2["validate_dir"] = split_dir
    # Evaluation must be deterministic even when inspecting the training view.
    ds = build_dataset(h2, visualize=False, train=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=ds.collate_batch_test)
    print("Evaluating on %s: %d frames" % (opt.split, len(ds)))

    model = train_utils.create_model(hypes)
    _, model = train_utils.load_saved_model(opt.model_dir, model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    use_amp = bool(torch.cuda.is_available() and not opt.no_amp)

    result_stat = {opt.iou_thresh: {"tp": [], "fp": [], "score": [], "gt": 0}}
    n_gt_present, n_pred_present = 0, 0
    inference_seconds = []

    with torch.inference_mode():
        for batch_data in loader:
            if batch_data is None:
                continue
            batch_data = train_utils.to_device(batch_data, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.cuda.amp.autocast(enabled=use_amp):
                output_dict = model(batch_data["ego"])
            if device.type == "cuda":
                torch.cuda.synchronize()
            inference_seconds.append(time.perf_counter() - started)
            pred_box_tensor, pred_score, gt_box_tensor = ds.post_process(
                batch_data, {"ego": output_dict})

            if gt_box_tensor is not None and gt_box_tensor.shape[0] > 0:
                n_gt_present += 1
            if pred_box_tensor is not None and pred_box_tensor.shape[0] > 0:
                n_pred_present += 1

            eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor,
                                       result_stat, opt.iou_thresh)

    stat = result_stat[opt.iou_thresh]
    tp, fp, gt = sum(stat["tp"]), sum(stat["fp"]), stat["gt"]
    recall = tp / gt if gt > 0 else float("nan")
    average_precision, _, _ = eval_utils.calculate_ap(result_stat, opt.iou_thresh)

    report = {
        "model_dir": opt.model_dir, "split": opt.split, "n_frames": len(ds),
        "iou_thresh": opt.iou_thresh,
        "frames_with_gt_box": n_gt_present, "frames_with_prediction": n_pred_present,
        "true_positives": tp, "false_positives": fp, "total_gt_boxes": gt,
        "recall_at_iou": recall, "average_precision": average_precision,
        "amp": use_amp,
        "mean_model_seconds": (sum(inference_seconds) / len(inference_seconds)
                               if inference_seconds else None),
        "note": ("Development metrics only. Final evaluation requires a new, "
                 "independent QCar trajectory configured as test_dir."),
    }
    print(json.dumps(report, indent=2))
    out = os.path.join(opt.model_dir, "eval_%s.json" % opt.split)
    json.dump(report, open(out, "w"), indent=2)
    print("\nwritten:", out)
    if opt.split == "test":
        print("\n*** This was the TEST split. Do not run it again for this model. ***")


if __name__ == "__main__":
    main()
