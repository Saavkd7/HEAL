"""Lean inference/evaluation for the QCar-real fine-tuning runs -- no open3d,
no visualisation, just decode + NMS + the recall@IoU metric this project has
used since session37 (target found within a given IoU of the true box).

Reuses HEAL's own decode+NMS path
(IntermediateheterFusionDataset.post_process -> VoxelPostprocessor.post_process,
the SAME code that generates training labels) and its own TP/FP counter
(opencood.utils.eval_utils.caluclate_tp_fp) rather than re-deriving IoU
matching from scratch -- one less place for a silent convention mismatch.

    python qcar/eval.py                                   # conf.json defaults
    python qcar/eval.py --model_dir opencood/logs/<run> --split validate

Modules are replayed from the run's resolved_hypes.json (_qcar_plugins,
_qcar_model_packages), so evaluation uses exactly what training used; runs
trained before those keys existed fall back to qcar/conf.json.

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

from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.utils import eval_utils

from qcar import config, registry


def parse_args():
    conf = config.load_conf(config.conf_path_from_argv())
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_dir", default=None,
                    help="directory produced by qcar/train.py (contains "
                         "resolved_hypes.json + a net_epoch_bestval_at*.pth; "
                         "default: conf.json eval_model_dir)")
    ap.add_argument("--split", choices=["train", "validate", "test"],
                    default=conf["eval_split"])
    ap.add_argument("--iou_thresh", type=float, default=conf["eval_iou_thresh"],
                    help="recall@IoU threshold (0.2 = convention since session37)")
    ap.add_argument("--amp", dest="amp", action="store_true",
                    help="CUDA mixed precision on (default: conf.json eval_amp)")
    ap.add_argument("--no_amp", dest="amp", action="store_false",
                    help="CUDA mixed precision off")
    ap.set_defaults(amp=conf["eval_amp"])
    config.add_module_args(ap)
    opt = ap.parse_args()
    opt.model_dir = (config.cli_path(opt.model_dir) if opt.model_dir
                     else conf.path_of("eval_model_dir"))
    return opt, conf


def main():
    opt, conf = parse_args()
    os.chdir(config.REPO_ROOT)
    # resolved_hypes.json is the FULLY resolved config train.py dumped
    # next to the checkpoints -- every override (Ncams, depth_supervision,
    # assignment_path, split dirs) already applied, so this is what actually
    # trained the model, not a re-derivation that could silently drift from it.
    hypes = json.load(open(os.path.join(opt.model_dir, "resolved_hypes.json")))
    config.resolve_modules(opt, hypes, conf)
    config.apply_overrides(hypes, conf["overrides"], opt.set)

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

    model = registry.create_model(hypes)
    _, model = train_utils.load_saved_model(opt.model_dir, model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    use_amp = bool(torch.cuda.is_available() and opt.amp)

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
            # HEAL's VoxelPostprocessor writes into fp32 anchor tensors; under
            # AMP the heads return fp16, so decode in fp32.
            output_dict = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
                           for k, v in output_dict.items()}
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
        "plugins": hypes["_qcar_plugins"],
        "fusion_method": hypes["model"]["args"].get("fusion_method"),
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
