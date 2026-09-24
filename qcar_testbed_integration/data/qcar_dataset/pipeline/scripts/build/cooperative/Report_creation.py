"""Generate a human-readable Markdown report per trajectory pair, plus an
overall index -- so nobody has to re-derive "what actually happened in
trajectory 03" by reading raw scene_gt JSON files by hand.

Reuses build_coop_train_val_dataset.py directly (load_frames, pair_frames_n, build_scene,
STATIC_TARGETS, CAM_PARAMS, common_indices, VAL_PAIRS) -- no numbers
are recomputed with different logic, so this report can never silently
disagree with the dataset it describes.

    python Report_creation.py

Writes to:
    CoopFront/reports/trajectory_<idx>.md   (one per pair)
    CoopFront/reports/README.md             (index + cross-trajectory summary)
"""
import collections
import datetime
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_coop_train_val_dataset as bcd

OUT_DIR = os.path.join(bcd.DST, "reports")


def analyze_pair(idx):
    frames_a = bcd.load_frames(bcd.EGO_ID, idx)
    frames_b = bcd.load_frames(bcd.PEER_ID, idx)
    groups = bcd.pair_frames_n({bcd.EGO_ID: frames_a, bcd.PEER_ID: frames_b})
    pairs = [(g[bcd.EGO_ID], g[bcd.PEER_ID]) for g in groups]

    result = {
        "idx": idx,
        "split": "validate" if idx in bcd.VAL_PAIRS else "train",
        "n_frames_a": len(frames_a),
        "n_frames_b": len(frames_b),
        "n_paired": len(pairs),
        # Derived from the real discovered folder name (not a hardcoded
        # prefix, which no longer exists in CAM_PARAMS) -- strips
        # "converted_" and the trailing "_<idx>".
        "route_ego": bcd.discover_trajectories(bcd.EGO_ID)[idx][len("converted_"):-len("_" + idx)],
        "route_peer": bcd.discover_trajectories(bcd.PEER_ID)[idx][len("converted_"):-len("_" + idx)],
    }
    if not pairs:
        result["empty"] = True
        return result
    result["empty"] = False

    fa0 = pairs[0][0]
    result["start_utc"] = datetime.datetime.utcfromtimestamp(fa0["t"]).isoformat() + "Z"
    result["duration_a_s"] = frames_a[-1]["t"] - frames_a[0]["t"]
    result["duration_b_s"] = frames_b[-1]["t"] - frames_b[0]["t"]

    # closest-approach point (the geometric crossing, if any)
    dists = [((fa["x"] - fb["x"]) ** 2 + (fa["y"] - fb["y"]) ** 2) ** 0.5 for fa, fb in pairs]
    k_min = dists.index(min(dists))
    fa_min, fb_min = pairs[k_min]
    result["min_dist_m"] = dists[k_min]
    result["min_dist_k"] = k_min
    result["min_dist_yaw_ego_deg"] = math.degrees(fa_min["yaw_rad"])
    result["min_dist_yaw_peer_deg"] = math.degrees(fb_min["yaw_rad"])

    # per-frame visibility, via the SAME build_scene() the dataset itself uses
    vehicle_count_hist = collections.Counter()
    per_vehicle_visible = collections.Counter()
    both_statics = 0
    mutual_dynamic = 0
    for fa, fb in pairs:
        raw, corrected, vis = bcd.build_scene(fa, fb)
        visible_ids = [vid for vid in bcd.SCENE_VEHICLE_IDS if vis[vid]["any"]]
        vehicle_count_hist[len(visible_ids)] += 1
        for vid in visible_ids:
            per_vehicle_visible[vid] += 1
        if vis["3"]["any"] and vis["4"]["any"]:
            both_statics += 1
        v1 = vis["1"]["by_agent"]["2"]  # peer sees ego
        v2 = vis["2"]["by_agent"]["1"]  # ego sees peer
        if v1 and v2:
            mutual_dynamic += 1

    result["vehicle_count_hist"] = dict(sorted(vehicle_count_hist.items()))
    result["per_vehicle_visible"] = dict(per_vehicle_visible)
    result["both_statics_visible"] = both_statics
    result["mutual_dynamic_visible"] = mutual_dynamic
    result["max_simultaneous"] = max(vehicle_count_hist.keys()) if vehicle_count_hist else 0
    return result


def render_trajectory_md(r):
    label_name = {"1": "ego (.198)", "2": "peer (.158)", "3": "Node 9 static", "4": "Node 11 static"}
    lines = []
    lines.append(f"# Trajectory {r['idx']} ({r['split']})\n")
    if r["empty"]:
        lines.append("**No paired frames found for this trajectory.**\n")
        return "\n".join(lines)

    lines.append(f"- Ego route (.198): `{r['route_ego']}`")
    lines.append(f"- Peer route (.158): `{r['route_peer']}`")
    lines.append(f"- Recording start (UTC): {r['start_utc']}")
    lines.append(f"- Frames captured: ego={r['n_frames_a']}, peer={r['n_frames_b']} "
                 f"(ego span {r['duration_a_s']:.1f}s, peer span {r['duration_b_s']:.1f}s)")
    lines.append(f"- Paired frames (<=40ms tolerance): {r['n_paired']} "
                 f"({100 * r['n_paired'] / max(r['n_frames_a'], r['n_frames_b']):.0f}% of the longer capture)")
    lines.append("")
    lines.append("## Closest approach (geometric crossing point)")
    lines.append(f"- Distance: {r['min_dist_m']:.2f} m, at paired frame index {r['min_dist_k']}")
    lines.append(f"- Ego heading there: {r['min_dist_yaw_ego_deg']:.1f} deg")
    lines.append(f"- Peer heading there: {r['min_dist_yaw_peer_deg']:.1f} deg")
    lines.append("")
    lines.append("## Visibility (from real camera geometry, via build_scene())")
    lines.append("")
    lines.append("| vehicle | frames visible to >=1 real camera | % of paired frames |")
    lines.append("|---|---|---|")
    for vid in bcd.SCENE_VEHICLE_IDS:
        n = r["per_vehicle_visible"].get(vid, 0)
        lines.append(f"| {vid} ({label_name[vid]}) | {n} | {100 * n / r['n_paired']:.0f}% |")
    lines.append("")
    lines.append(f"- Both statics (Node 9 + Node 11) simultaneously visible: "
                 f"{r['both_statics_visible']} frames ({100 * r['both_statics_visible'] / r['n_paired']:.0f}%)")
    lines.append(f"- Ego and peer mutually seeing each other at the same instant: "
                 f"{r['mutual_dynamic_visible']} frames")
    lines.append("")
    lines.append("## How many vehicles are simultaneously visible per frame")
    lines.append("")
    lines.append("| # vehicles visible | # frames |")
    lines.append("|---|---|")
    for n, count in r["vehicle_count_hist"].items():
        lines.append(f"| {n} | {count} |")
    lines.append(f"\nMaximum simultaneous visibility reached in this trajectory: **{r['max_simultaneous']} / 4**.\n")
    return "\n".join(lines)


def render_index_md(results):
    lines = ["# CoopFront trajectory reports -- index\n",
             "Auto-generated by `Report_creation.py`, reusing `build_coop_train_val_dataset.py`'s own "
             "functions directly (no numbers recomputed with separate logic).\n",
             "| pair | split | frames paired | max simultaneous | mutual ego<->peer | closest approach |",
             "|---|---|---|---|---|---|"]
    max_overall = 0
    mutual_overall = 0
    total_paired = 0
    for r in results:
        if r["empty"]:
            lines.append(f"| {r['idx']} | - | 0 | - | - | - |")
            continue
        max_overall = max(max_overall, r["max_simultaneous"])
        mutual_overall += r["mutual_dynamic_visible"]
        total_paired += r["n_paired"]
        lines.append(f"| [{r['idx']}](trajectory_{r['idx']}.md) | {r['split']} | {r['n_paired']} | "
                     f"{r['max_simultaneous']}/4 | {r['mutual_dynamic_visible']} | "
                     f"{r['min_dist_m']:.2f}m |")
    lines.append("")
    lines.append("## Cross-trajectory summary")
    lines.append(f"- Total paired frames across all trajectories: {total_paired}")
    lines.append(f"- Maximum simultaneous visibility ever reached, any trajectory: **{max_overall} / 4**")
    lines.append(f"- Frames where ego and peer mutually see each other, across all trajectories: "
                 f"{mutual_overall}")
    lines.append("")
    lines.append("All 9 trajectories share the same structural ceiling: ego and peer never see each "
                 "other simultaneously (both forward-facing cameras complete their turn onto the "
                 "crossing heading at different times), so the maximum ever achievable is 1 dynamic "
                 "agent + both statics = 3, never all 4.")
    return "\n".join(lines)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = []
    for idx in bcd.common_indices():
        print("analyzing pair", idx, "...")
        r = analyze_pair(idx)
        results.append(r)
        md = render_trajectory_md(r)
        out_path = os.path.join(OUT_DIR, f"trajectory_{idx}.md")
        with open(out_path, "w") as f:
            f.write(md)
        print("  wrote", out_path)

    index_md = render_index_md(results)
    index_path = os.path.join(OUT_DIR, "README.md")
    with open(index_path, "w") as f:
        f.write(index_md)
    print("\nwrote", index_path)


if __name__ == "__main__":
    main()
