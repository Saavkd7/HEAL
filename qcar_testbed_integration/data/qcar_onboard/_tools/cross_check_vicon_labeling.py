#!/usr/bin/env python3
"""Check the ego/target Vicon labeling bug against the 9 CoopFront trajectories.

`bag_to_dataset_rosbags.py` documents (but does not fix) a known bug: the
QCar's own Vicon bridge sometimes mislabels which car is "ego" and which is
"target". This script answers whether that bug actually touched the 9
trajectories `build_coop_train_val_dataset.py` uses (`_00`-`_08`).

REAL ARCHITECTURE (confirmed 2026-09-18 by reading both bags' raw
`/qcar/vicon` messages directly, not assumed): this is NOT two cars each
self-reporting "I am ego, my peer is target". It's ONE shared broadcast --
every message on `/qcar/vicon`, in BOTH bags, carries the SAME two absolute
positions under `data.ego` / `data.target`. Verified directly: at matching
wall-clock times, .198's raw "ego" field equals .158's raw "ego" field
(sub-mm), and same for "target" -- i.e. "ego"/"target" are fixed GLOBAL
roles for the whole capture session, not a per-car self/peer distinction.
An earlier version of this script assumed the self/peer model and reported
every trajectory as "MISLABELED" on that basis -- that verdict was wrong,
not a real finding. Left in `--diagnostic` mode below for reference only.

THE CHECK THAT ACTUALLY MATTERS: since each car's `bag_to_dataset_rosbags.py`
run picks ONE field (named by `vicon_pose_key`) to save as that car's own
`ego_vicon_pose.json`, the only way the bug bites is if a car's resolved key
ever flips between sessions -- e.g. if .198 (normally "ego") resolved
"target" for some trajectory, its saved `ego_vicon_pose.json` would silently
hold .158's position instead of its own. This script's default mode checks
exactly that: `resolved_vicon_pose_key` per trajectory per car, looking for
any deviation from that car's majority/expected role.

Result as of 2026-09-18: .198 resolves "ego" and .158 resolves "target" in
ALL 9 trajectories (`_00`-`_08`), zero exceptions -- no evidence the
mislabeling bug affected any trajectory CoopFront actually uses.

Limitation: this cannot catch a bug where a car's role was wrong from the
very first trajectory it ever recorded (i.e. consistently "wrong" in a way
that never flips) -- there is no fully external ground truth here, only
cross-session consistency.

Usage:
    <heal38 env python> cross_check_vicon_labeling.py
    <heal38 env python> cross_check_vicon_labeling.py --trajectories 00 03 07
    <heal38 env python> cross_check_vicon_labeling.py --diagnostic   # raw ego/target
                                                                       # distance table,
                                                                       # for inspecting the
                                                                       # shared-broadcast
                                                                       # claim itself, not
                                                                       # for bug-hunting

Needs the `rosbags` package (present in the `heal38` pyenv env this project
otherwise uses, confirmed 2026-09-18):
    ~/.pyenv/versions/heal38/bin/python cross_check_vicon_labeling.py
"""
import argparse
import json
import os

import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))  # _tools/ -> qcar_onboard/ -> data/ -> qcar_testbed_integration/
DEFAULT_CONF = os.path.join(HERE, "conf.json")


class _Conf(dict):
    """conf.json contents. It is THE source of every default -- nothing is
    hardcoded in this script; a CLI flag only overrides a key for one run.
    A missing key is a clear error, never a silent fallback."""

    def __init__(self, path, data):
        dict.__init__(self, data)
        self.path = path

    def __missing__(self, key):
        raise SystemExit("%s has no %r key -- add it there" % (self.path, key))

    def path_of(self, key):
        """Path-valued key: relative paths resolve against the
        qcar_testbed_integration/ root (REPO_ROOT); null stays None."""
        v = self[key]
        return v if not v or os.path.isabs(v) else os.path.join(REPO_ROOT, v)


def _load_conf(path):
    if not os.path.exists(path):
        raise SystemExit("conf.json not found: %s" % path)
    with open(path) as f:
        return _Conf(path, json.load(f))


_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--conf", default=DEFAULT_CONF)
_conf = _load_conf(_pre.parse_known_args()[0].conf)

TYPESTORE = get_typestore(Stores.ROS1_NOETIC)
VICON_TOPIC = _conf["cross_check_vicon_topic"]

# Every value below comes from conf.json (cross_check_* keys). Only
# trajectories 00-08 have a surviving .bag (09's .bag was never kept).
ONBOARD = _conf.path_of("cross_check_onboard_root")  # replaced by --onboard-root in main()
CARS = _conf["cross_check_cars"]
TRAJECTORIES = _conf["cross_check_trajectories"]
DEFAULT_TOLERANCE_SEC = _conf["cross_check_tolerance_sec"]  # a bit over one Vicon period (~48.7 Hz => ~20.5ms)


def car_dir(ip):
    return os.path.join(ONBOARD, "192.168.1.%s" % ip)


def bag_path(ip, traj):
    """Raw .bag lives under <car>/bags/<trajectory>/ (separate from the
    converted per-frame output, which lives under <car>/dataset/)."""
    prefix = CARS[ip]["prefix"]
    return os.path.join(car_dir(ip), "bags", prefix + traj, prefix + traj + ".bag")


def converted_metadata_path(ip, traj):
    """Converted-frame metadata lives under <car>/dataset/converted_<trajectory>/."""
    prefix = CARS[ip]["prefix"]
    return os.path.join(car_dir(ip), "dataset", "converted_" + prefix + traj, "metadata.json")


# ---------------------------------------------------------------------------
# Primary check: resolved_vicon_pose_key consistency across trajectories
# ---------------------------------------------------------------------------

def resolved_key(ip, traj):
    path = converted_metadata_path(ip, traj)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f).get("resolved_vicon_pose_key")


def run_consistency_check(trajectories):
    per_car = {ip: {} for ip in CARS}
    for ip in CARS:
        for traj in trajectories:
            per_car[ip][traj] = resolved_key(ip, traj)

    print("%-6s %10s %10s" % ("traj", "198", "158"))
    print("-" * 28)
    for traj in trajectories:
        print("%-6s %10s %10s" % (traj, per_car["198"][traj], per_car["158"][traj]))

    flagged = []
    for ip in CARS:
        values = [v for v in per_car[ip].values() if v is not None]
        if not values:
            continue
        majority = max(set(values), key=values.count)
        for traj, v in per_car[ip].items():
            if v is not None and v != majority:
                flagged.append((ip, traj, v, majority))

    print()
    if flagged:
        print("FLAGGED -- role differs from this car's majority resolved key:")
        for ip, traj, v, majority in flagged:
            print("  .%s trajectory %s resolved '%s', expected '%s' (majority) "
                  "-- ego_vicon_pose.json for this frame likely holds the WRONG car's position"
                  % (ip, traj, v, majority))
    else:
        print("No deviations: each car resolved the same Vicon role in every "
              "trajectory checked. No evidence the mislabeling bug affected "
              "this set.")
    return flagged


# ---------------------------------------------------------------------------
# Diagnostic-only: raw ego/target distances, to inspect the shared-broadcast
# claim itself (NOT a bug detector -- see module docstring).
# ---------------------------------------------------------------------------

def read_vicon_stream(path):
    entries = []
    with Reader(path) as reader:
        connections = [c for c in reader.connections if c.topic == VICON_TOPIC]
        if not connections:
            raise RuntimeError("No %s topic in %s" % (VICON_TOPIC, path))
        for connection, _, rawdata in reader.messages(connections=connections):
            msg = TYPESTORE.deserialize_ros1(rawdata, connection.msgtype)
            try:
                envelope = json.loads(msg.data)
            except (TypeError, ValueError):
                continue
            payload = envelope.get("data", envelope)
            ego = payload.get("ego")
            target = payload.get("target")
            if not ego or not target:
                continue
            if not ego.get("valid") or not target.get("valid"):
                continue
            wall_time = ego.get("wall_time_unix", target.get("wall_time_unix"))
            if wall_time is None:
                continue
            entries.append({
                "wall_time": float(wall_time),
                "ego": np.array([ego["x"], ego["y"], ego["z"]], dtype=np.float64),
                "target": np.array([target["x"], target["y"], target["z"]], dtype=np.float64),
            })
    entries.sort(key=lambda e: e["wall_time"])
    return entries


def nearest_match(sorted_entries, wall_times, t, tolerance):
    idx = np.searchsorted(wall_times, t)
    best, best_dt = None, tolerance
    for i in (idx - 1, idx):
        if 0 <= i < len(sorted_entries):
            dt = abs(sorted_entries[i]["wall_time"] - t)
            if dt <= best_dt:
                best, best_dt = sorted_entries[i], dt
    return best


def run_diagnostic(trajectories, tolerance):
    header = "%-6s %6s %6s %6s %11s %11s %11s %11s" % (
        "traj", "n198", "n158", "match", "198e~158e", "198e~158t",
        "198t~158e", "198t~158t")
    print(header)
    print("-" * len(header))
    print("(near-zero on the ee/tt columns is EXPECTED -- it confirms the "
          "shared-broadcast model, it is not a mislabeling verdict)")
    for traj in trajectories:
        p198, p158 = bag_path("198", traj), bag_path("158", traj)
        if not os.path.exists(p198) or not os.path.exists(p158):
            print("%-6s SKIPPED (missing .bag)" % traj)
            continue
        s198, s158 = read_vicon_stream(p198), read_vicon_stream(p158)
        times_158 = np.array([e["wall_time"] for e in s158])
        diffs = {"ee": [], "et": [], "te": [], "tt": []}
        for e in s198:
            m = nearest_match(s158, times_158, e["wall_time"], tolerance)
            if m is None:
                continue
            diffs["ee"].append(np.linalg.norm(e["ego"] - m["ego"]))
            diffs["et"].append(np.linalg.norm(e["ego"] - m["target"]))
            diffs["te"].append(np.linalg.norm(e["target"] - m["ego"]))
            diffs["tt"].append(np.linalg.norm(e["target"] - m["target"]))
        n = len(diffs["ee"])
        if n == 0:
            print("%-6s %6d %6d %6d  no matched samples within tolerance" % (
                traj, len(s198), len(s158), n))
            continue
        print("%-6s %6d %6d %6d %11.4f %11.4f %11.4f %11.4f" % (
            traj, len(s198), len(s158), n,
            np.mean(diffs["ee"]), np.mean(diffs["et"]),
            np.mean(diffs["te"]), np.mean(diffs["tt"])))


def main():
    global ONBOARD
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--conf", default=DEFAULT_CONF,
                        help="conf.json to read defaults from (default: %s)" % DEFAULT_CONF)
    parser.add_argument("--onboard-root", default=ONBOARD,
                        help="folder holding one 192.168.1.<ip>/ per car (default: conf.json's cross_check_onboard_root)")
    parser.add_argument("--trajectories", nargs="*", default=TRAJECTORIES)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE_SEC)
    parser.add_argument("--diagnostic", action="store_true",
                         help="print raw ego/target distance table instead of "
                              "the role-consistency verdict")
    args = parser.parse_args()
    ONBOARD = args.onboard_root

    if args.diagnostic:
        run_diagnostic(args.trajectories, args.tolerance)
    else:
        run_consistency_check(args.trajectories)


if __name__ == "__main__":
    main()
