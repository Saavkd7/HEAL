"""Why do some fusions detect (IoU 0.2) but localize badly (IoU 0.5)?

For each zoo method, on the validate split, in fp32:

  1. localization error of every GT box matched by a prediction at IoU >= 0.2:
     center error (cm, real scale), mean bias vector (systematic shift?),
     yaw error (deg), length/width ratio pred/GT, and the fraction of those
     matches that also reach IoU >= 0.5;
  2. the same split by WHO could see the target: 'ego' = the ego-only pass
     also finds it (IoU >= 0.2), 'peer_only' = only found with the peer's
     features -- a misaligned peer (e.g. uncalibrated extrinsic) shows up as
     larger / biased error on peer_only targets;
  3. AP@0.2 / AP@0.5 of the LAST epoch (training_state_last.pth of the source
     run) vs the checkpoint picked by lowest validation loss.

Model space is x10 the real world in x/y (1 unit = 10 cm).

    python qcar/analyze_localization.py                 # conf.json analyze_* keys
    python qcar/analyze_localization.py --methods att pyramid
"""
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from opencood.tools import train_utils
from opencood.utils import common_utils, eval_utils

from qcar import compare, config, registry

CM_PER_UNIT = 10.0


def parse_args():
    conf = config.load_conf(config.conf_path_from_argv())
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conf", default=config.DEFAULT_CONF)
    ap.add_argument("--zoo_dir", default=None, help="default: conf.json compare_zoo_dir")
    ap.add_argument("--methods", nargs="*", default=conf["analyze_methods"])
    ap.add_argument("--split", default=conf["compare_split"])
    ap.add_argument("--out", default=None, help="default: <zoo_dir>/LOCALIZATION.json")
    opt = ap.parse_args()
    opt.zoo_dir = (config.cli_path(opt.zoo_dir) if opt.zoo_dir
                   else conf.path_of("compare_zoo_dir"))
    opt.out = (config.cli_path(opt.out) if opt.out
               else os.path.join(opt.zoo_dir, "LOCALIZATION.json"))
    return opt, conf


def geometry(corners):
    """(N,8,3) corners -> center xy, yaw (mod pi), length, width from the bottom face."""
    bottom = corners[:, :4, :2]
    center = bottom.mean(axis=1)
    e1 = bottom[:, 1] - bottom[:, 0]
    e2 = bottom[:, 2] - bottom[:, 1]
    n1, n2 = np.linalg.norm(e1, axis=1), np.linalg.norm(e2, axis=1)
    long_edge = np.where((n1 >= n2)[:, None], e1, e2)
    yaw = np.arctan2(long_edge[:, 1], long_edge[:, 0]) % np.pi
    return center, yaw, np.maximum(n1, n2), np.minimum(n1, n2)


def best_iou(gt_corners, pred_corners):
    """For each GT, the index and IoU of its best-overlapping prediction."""
    if len(pred_corners) == 0:
        return np.full(len(gt_corners), -1), np.zeros(len(gt_corners))
    preds = common_utils.convert_format(pred_corners)
    idx, ious = [], []
    for g in common_utils.convert_format(gt_corners):
        iou = common_utils.compute_iou(g, preds)
        idx.append(int(np.argmax(iou)))
        ious.append(float(np.max(iou)))
    return np.array(idx), np.array(ious)


def predict(model, ds, device):
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=ds.collate_batch_test)
    outs = []
    with torch.inference_mode():
        for batch in loader:
            if batch is None:
                outs.append(None)
                continue
            batch = train_utils.to_device(batch, device)
            out = model(batch["ego"])
            pred, score, gt = ds.post_process(batch, {"ego": out})
            np_ = lambda t: t.cpu().numpy() if t is not None else np.zeros((0, 8, 3))
            outs.append((np_(pred), None if score is None else score.cpu().numpy(), np_(gt),
                         pred, score, gt))
    return outs


def ap_at(outs, iou):
    stat = {iou: {"tp": [], "fp": [], "score": [], "gt": 0}}
    for o in outs:
        if o is not None:
            eval_utils.caluclate_tp_fp(o[3], o[4], o[5], stat, iou)
    return float(eval_utils.calculate_ap(stat, iou)[0]) if stat[iou]["gt"] else float("nan")


def summarize(rows):
    if not rows:
        return {"n": 0}
    r = np.array(rows)  # dx, dy, yaw_err, l_ratio, w_ratio, iou
    err = np.hypot(r[:, 0], r[:, 1]) * CM_PER_UNIT
    return {
        "n": int(len(r)),
        "center_err_cm_median": float(np.median(err)),
        "center_err_cm_p90": float(np.percentile(err, 90)),
        "bias_cm": [float(r[:, 0].mean() * CM_PER_UNIT), float(r[:, 1].mean() * CM_PER_UNIT)],
        "yaw_err_deg_median": float(np.degrees(np.median(r[:, 2]))),
        "length_ratio_median": float(np.median(r[:, 3])),
        "width_ratio_median": float(np.median(r[:, 4])),
        "frac_iou_ge_0.5": float(np.mean(r[:, 5] >= 0.5)),
    }


def analyze(method_dir, split, conf):
    hypes = json.load(open(os.path.join(method_dir, "resolved_hypes.json")))
    config.resolve_modules(argparse.Namespace(plugins=None, model_packages=None,
                                              loss_packages=None), hypes, conf)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = registry.create_model(hypes)
    _, model = train_utils.load_saved_model(method_dir, model)
    model.to(device).eval()

    ds = compare.build_split(hypes, split)
    ds_ego = compare.build_split(hypes, split, conf["compare_ego_only_comm_range"])
    coop, ego = predict(model, ds, device), predict(model, ds_ego, device)

    rows = {"all": [], "ego": [], "peer_only": []}
    for c, e in zip(coop, ego):
        if c is None or e is None or len(c[2]) == 0:
            continue
        gt = c[2]
        idx, iou = best_iou(gt, c[0])
        _, iou_ego = best_iou(gt, e[0])
        g_c, g_yaw, g_l, g_w = geometry(gt)
        for k in np.where(iou >= 0.2)[0]:
            p_c, p_yaw, p_l, p_w = geometry(c[0][idx[k]:idx[k] + 1])
            d = np.abs(p_yaw[0] - g_yaw[k]) % np.pi
            row = [p_c[0, 0] - g_c[k, 0], p_c[0, 1] - g_c[k, 1], min(d, np.pi - d),
                   p_l[0] / g_l[k], p_w[0] / g_w[k], iou[k]]
            rows["all"].append(row)
            rows["ego" if iou_ego[k] >= 0.2 else "peer_only"].append(row)

    result = {k: summarize(v) for k, v in rows.items()}
    result["ap_best"] = {"0.2": ap_at(coop, 0.2), "0.5": ap_at(coop, 0.5)}

    source = open(os.path.join(method_dir, "SOURCE_RUN.txt")).read().split()[2]
    last = os.path.join(config.repo_path(source), "training_state_last.pth")
    if os.path.isfile(last):
        state = torch.load(last, map_location="cpu")
        model.load_state_dict(state["model_state_dict"])
        model.to(device).eval()
        last_out = predict(model, ds, device)
        result["ap_last_epoch"] = {"0.2": ap_at(last_out, 0.2), "0.5": ap_at(last_out, 0.5),
                                   "epoch": int(state["next_epoch"])}
    del model
    torch.cuda.empty_cache()
    return result


def main():
    opt, conf = parse_args()
    os.chdir(config.REPO_ROOT)
    report = {}
    for m in opt.methods:
        print("== %s" % m, flush=True)
        report[m] = analyze(os.path.join(opt.zoo_dir, m), opt.split, conf)
        print(json.dumps(report[m]), flush=True)
    with open(opt.out, "w") as f:
        json.dump(report, f, indent=2)
    print("\n%-11s %6s %6s %7s %7s %9s %9s %14s %7s %6s %6s %13s" % (
        "method", "AP.2", "AP.5", "err_cm", "p90_cm", "egoErr", "peerErr",
        "bias_cm(x,y)", "yaw_deg", "L_rat", "W_rat", "AP.5 last"))
    for m, r in report.items():
        a, e, p = r["all"], r["ego"], r["peer_only"]
        print("%-11s %6.3f %6.3f %7.1f %7.1f %9s %9s %14s %7.1f %6.2f %6.2f %13s" % (
            m, r["ap_best"]["0.2"], r["ap_best"]["0.5"], a["center_err_cm_median"],
            a["center_err_cm_p90"],
            "%.1f(%d)" % (e["center_err_cm_median"], e["n"]) if e["n"] else "-",
            "%.1f(%d)" % (p["center_err_cm_median"], p["n"]) if p["n"] else "-",
            "(%.1f,%.1f)" % tuple(a["bias_cm"]), a["yaw_err_deg_median"],
            a["length_ratio_median"], a["width_ratio_median"],
            "%.3f@ep%d" % (r["ap_last_epoch"]["0.5"], r["ap_last_epoch"]["epoch"])
            if "ap_last_epoch" in r else "-"))


if __name__ == "__main__":
    main()
