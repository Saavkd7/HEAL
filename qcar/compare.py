"""Which fusion fuses best on QCar? Statistical comparison of the model zoo.

For every trained method in the zoo (checkpoints/qcar/zoo/<method>/) it runs
the validate split twice on the SAME frames:

  coop      the normal cooperative pass (ego + peer features fused)
  ego_only  the peer is dropped from the input (comm_range -> ~0), but the
            detections are scored against the COOPERATIVE ground truth of the
            same frame -- dropping the peer through comm_range also drops its
            labels in HEAL, which would silently shrink the GT and inflate recall.

  fusion gain = AP(coop) - AP(ego_only): what the fusion itself contributes.

WHY A BOOTSTRAP, AND WHY BY BLOCKS
------------------------------------
Validate frames are consecutive 10 Hz frames of a few whole trajectories, so
neighbouring frames are nearly duplicates; treating 237 frames as 237
independent samples would make every difference look significant. The
confidence intervals therefore come from a stratified block bootstrap:
blocks of `compare_block_frames` consecutive frames, resampled with
replacement inside each trajectory, all methods evaluated on the SAME
resample (paired). A method is called better than another only when the 95%
interval of their AP difference excludes zero.

Caveat printed in the report: validate also picked each run's best epoch
(lowest val loss), a mild optimism shared by every method.

    python qcar/compare.py                 # everything from conf.json compare_* keys
    python qcar/compare.py --methods att max v2xvit --bootstrap 2000
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.utils import eval_utils

from qcar import config, registry

MODES = ("coop", "ego_only")


def parse_args():
    conf = config.load_conf(config.conf_path_from_argv())
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conf", default=config.DEFAULT_CONF)
    ap.add_argument("--zoo_dir", default=None,
                    help="default: conf.json compare_zoo_dir")
    ap.add_argument("--methods", nargs="*", default=None,
                    help="default: every trained method found in the zoo dir")
    ap.add_argument("--split", choices=["train", "validate", "test"],
                    default=conf["compare_split"])
    ap.add_argument("--iou", type=float, nargs="+", default=conf["compare_iou"])
    ap.add_argument("--bootstrap", type=int, default=conf["compare_bootstrap"])
    ap.add_argument("--block_frames", type=int, default=conf["compare_block_frames"])
    ap.add_argument("--ego_only_comm_range", type=float,
                    default=conf["compare_ego_only_comm_range"])
    ap.add_argument("--seed", type=int, default=conf["compare_seed"])
    ap.add_argument("--out", default=None, help="default: <zoo_dir>")
    opt = ap.parse_args()
    opt.zoo_dir = (config.cli_path(opt.zoo_dir) if opt.zoo_dir
                   else conf.path_of("compare_zoo_dir"))
    opt.out = config.cli_path(opt.out) if opt.out else opt.zoo_dir
    return opt, conf


def trained_methods(zoo_dir, wanted):
    found = sorted(d for d in os.listdir(zoo_dir)
                   if os.path.isfile(os.path.join(zoo_dir, d, "resolved_hypes.json"))
                   and any(f.startswith("net_epoch_bestval_at")
                           for f in os.listdir(os.path.join(zoo_dir, d))))
    if wanted:
        missing = [m for m in wanted if m not in found]
        if missing:
            raise SystemExit("not trained / not in %s: %s" % (zoo_dir, missing))
        return list(wanted)
    return found


def build_split(hypes, split, comm_range=None):
    h = json.loads(json.dumps(hypes))
    key = {"train": "root_dir", "validate": "validate_dir", "test": "test_dir"}[split]
    if not h.get(key):
        raise SystemExit("%s not configured in this run" % key)
    h["validate_dir"] = h[key]
    if comm_range is not None:
        h["comm_range"] = comm_range
    return build_dataset(h, visualize=False, train=False)


def frame_trajectory(ds, n_frames):
    """len_record holds the cumulative frame count at the end of each scenario."""
    ends = list(ds.len_record)
    names = [os.path.basename(str(p)) for p in getattr(ds, "scenario_folders", [])] \
        or ["traj%d" % i for i in range(len(ends))]
    traj = np.zeros(n_frames, dtype=int)
    start = 0
    for i, end in enumerate(ends):
        traj[start:end] = i
        start = end
    return traj, names


def run_method(method_dir, split, iou_list, ego_only_comm_range, conf):
    """Per-frame detection records for both modes, scored against coop GT."""
    hypes = json.load(open(os.path.join(method_dir, "resolved_hypes.json")))
    no_cli = argparse.Namespace(plugins=None, model_packages=None, loss_packages=None)
    config.resolve_modules(no_cli, hypes, conf)

    model = registry.create_model(hypes)
    _, model = train_utils.load_saved_model(method_dir, model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    ds_coop = build_split(hypes, split)
    ds_ego = build_split(hypes, split, ego_only_comm_range)
    assert len(ds_coop) == len(ds_ego)

    def predict(ds):
        loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=ds.collate_batch_test)
        outs = []
        with torch.inference_mode():
            for batch in loader:
                if batch is None:
                    outs.append(None)
                    continue
                batch = train_utils.to_device(batch, device)
                # fp32 for every method (CoAlign overflows fp16), so AMP is not a variable
                with torch.cuda.amp.autocast(enabled=False):
                    out = model(batch["ego"])
                outs.append(ds.post_process(batch, {"ego": out}))
        return outs

    coop = predict(ds_coop)
    ego = predict(ds_ego)

    records = {mode: {iou: [] for iou in iou_list} for mode in MODES}
    for frame in range(len(coop)):
        if coop[frame] is None or ego[frame] is None:
            for mode in MODES:
                for iou in iou_list:
                    records[mode][iou].append(([], [], [], 0))
            continue
        gt = coop[frame][2]
        for mode, (pred, score, _) in (("coop", coop[frame]), ("ego_only", ego[frame])):
            for iou in iou_list:
                stat = {iou: {"tp": [], "fp": [], "score": [], "gt": 0}}
                eval_utils.caluclate_tp_fp(pred, score, gt, stat, iou)
                s = stat[iou]
                records[mode][iou].append((s["tp"], s["fp"], s["score"], s["gt"]))
    traj, names = frame_trajectory(ds_coop, len(coop))
    del model
    torch.cuda.empty_cache()
    return records, traj, names


def ap_of(frame_records, frames):
    tp, fp, score, gt = [], [], [], 0
    for f in frames:
        t, p, s, g = frame_records[f]
        tp += list(t); fp += list(p); score += [float(x) for x in s]; gt += g
    if gt == 0:
        return float("nan"), float("nan")
    if not score:
        return 0.0, 0.0
    ap, _, _ = eval_utils.calculate_ap({0: {"tp": tp, "fp": fp, "score": score, "gt": gt}}, 0)
    return float(ap), float(sum(tp)) / gt


def block_resamples(traj, block, n_boot, rng):
    """Stratified moving-block bootstrap index sets (paired across methods)."""
    blocks_by_traj = []
    for t in np.unique(traj):
        idx = np.where(traj == t)[0]
        blocks_by_traj.append([idx[i:i + block] for i in range(0, len(idx), block)])
    for _ in range(n_boot):
        frames = []
        for blocks in blocks_by_traj:
            for j in rng.integers(0, len(blocks), size=len(blocks)):
                frames.extend(blocks[j])
        yield frames


def ci(values):
    v = np.asarray([x for x in values if np.isfinite(x)])
    if len(v) == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))


def main():
    opt, conf = parse_args()
    os.chdir(config.REPO_ROOT)
    methods = trained_methods(opt.zoo_dir, opt.methods)
    if len(methods) < 2:
        raise SystemExit("need at least 2 trained methods, found %s" % methods)
    print("comparing:", methods)

    records, traj, names = {}, None, None
    for m in methods:
        t0 = time.time()
        records[m], traj_m, names = run_method(os.path.join(opt.zoo_dir, m), opt.split,
                                               opt.iou, opt.ego_only_comm_range, conf)
        if traj is None:
            traj = traj_m
        elif len(traj_m) != len(traj) or (traj_m != traj).any():
            raise SystemExit("%s evaluated a different frame set" % m)
        print("  %-11s %.0fs" % (m, time.time() - t0))

    all_frames = np.arange(len(traj))
    rng = np.random.default_rng(opt.seed)
    resamples = list(block_resamples(traj, opt.block_frames, opt.bootstrap, rng))
    n_blocks = sum(int(np.ceil((traj == t).sum() / opt.block_frames)) for t in np.unique(traj))

    report = {"methods": methods, "split": opt.split, "frames": int(len(traj)),
              "trajectories": {names[t] if t < len(names) else str(t): int((traj == t).sum())
                               for t in np.unique(traj)},
              "block_frames": opt.block_frames, "independent_blocks": n_blocks,
              "bootstrap": opt.bootstrap, "seed": opt.seed, "iou": {}}

    for iou in opt.iou:
        res = {}
        point = {(m, mode): ap_of(records[m][mode][iou], all_frames)
                 for m in methods for mode in MODES}
        boot = {(m, mode): [] for m in methods for mode in MODES}
        for frames in resamples:
            for m in methods:
                for mode in MODES:
                    boot[(m, mode)].append(ap_of(records[m][mode][iou], frames)[0])
        boot = {k: np.array(v) for k, v in boot.items()}

        coop_matrix = np.stack([boot[(m, "coop")] for m in methods])  # (M, B)
        best_idx = np.nanargmax(coop_matrix, axis=0)
        leader = max(methods, key=lambda m: point[(m, "coop")][0])
        for i, m in enumerate(methods):
            per_traj = {}
            for t in np.unique(traj):
                per_traj[names[t] if t < len(names) else str(t)] = ap_of(
                    records[m]["coop"][iou], np.where(traj == t)[0])[0]
            gain = boot[(m, "coop")] - boot[(m, "ego_only")]
            diff = boot[(leader, "coop")] - boot[(m, "coop")]
            res[m] = {
                "ap_coop": point[(m, "coop")][0], "ap_coop_ci": ci(boot[(m, "coop")]),
                "recall_coop": point[(m, "coop")][1],
                "ap_ego_only": point[(m, "ego_only")][0],
                "ap_ego_only_ci": ci(boot[(m, "ego_only")]),
                "fusion_gain": point[(m, "coop")][0] - point[(m, "ego_only")][0],
                "fusion_gain_ci": ci(gain),
                "gap_to_leader": point[(leader, "coop")][0] - point[(m, "coop")][0],
                "gap_to_leader_ci": ci(diff),
                "p_best": float(np.mean(best_idx == i)),
                "ap_coop_per_trajectory": per_traj,
            }
        # the "top group": methods whose gap to the leader is not significant
        top = [m for m in methods
               if m == leader or not (res[m]["gap_to_leader_ci"][0] > 0)]
        report["iou"][str(iou)] = {"leader": leader, "statistically_tied_with_leader": top,
                                   "methods": res}

    os.makedirs(opt.out, exist_ok=True)
    with open(os.path.join(opt.out, "COMPARISON.json"), "w") as f:
        json.dump(report, f, indent=2)
    write_markdown(report, os.path.join(opt.out, "COMPARISON.md"))
    print(open(os.path.join(opt.out, "COMPARISON.md")).read())


def fmt_ci(v, c):
    return "%.3f [%.3f, %.3f]" % (v, c[0], c[1])


def write_markdown(report, path):
    lines = ["# QCar fusion comparison", "",
             "Split `%s`: %d frames from %s; %d-frame blocks -> %d independent blocks; "
             "%d paired block-bootstrap resamples (seed %d). 95%% intervals in brackets." % (
                 report["split"], report["frames"], report["trajectories"],
                 report["block_frames"], report["independent_blocks"],
                 report["bootstrap"], report["seed"]),
             "",
             "Caveat: validate also selected each run's best epoch (shared mild optimism); "
             "no independent test trajectory exists yet.", ""]
    for iou, block in report["iou"].items():
        res = block["methods"]
        order = sorted(res, key=lambda m: -res[m]["ap_coop"])
        lines += ["## IoU %s" % iou, "",
                  "Leader: **%s**. Not significantly worse than the leader: %s." % (
                      block["leader"], ", ".join(block["statistically_tied_with_leader"])), "",
                  "| method | AP coop | AP ego-only | fusion gain | gap to leader | P(best) | AP per trajectory |",
                  "|---|---|---|---|---|---|---|"]
        for m in order:
            r = res[m]
            lines.append("| %s | %s | %s | %s | %s | %.2f | %s |" % (
                m, fmt_ci(r["ap_coop"], r["ap_coop_ci"]),
                fmt_ci(r["ap_ego_only"], r["ap_ego_only_ci"]),
                fmt_ci(r["fusion_gain"], r["fusion_gain_ci"]),
                fmt_ci(r["gap_to_leader"], r["gap_to_leader_ci"]),
                r["p_best"],
                ", ".join("%s %.3f" % kv for kv in r["ap_coop_per_trajectory"].items())))
        lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
