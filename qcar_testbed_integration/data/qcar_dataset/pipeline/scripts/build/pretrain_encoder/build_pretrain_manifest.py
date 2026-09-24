"""Filter the 101-scenario collision corpus down to usable front-camera frames.

Source: conf.json's src_root (the 101-scenario corpus on external media)
Captured with qcar-52775 (192.168.1.198) per Kevin, front camera only.

A frame is kept only if:
  - front.png exists and is not the black warm-up frame (mean pixel > BLACK_THRESHOLD)
  - ego_vicon_pose.json exists and valid == 1.0
  - target_vicon_pose.json exists and valid == 1.0 (need ego pose always; target
    validity just tells us whether this frame is a positive or a clean negative,
    both are kept -- negatives are legitimate training signal)

Writes one manifest JSON with, per scenario: total frames seen, kept, dropped
(with a reason breakdown), and the list of kept frame ids with a `has_target`
flag. Does not touch calibration or scale -- that is the next script.
"""
from __future__ import annotations  # lets list[str]-style hints run on Python 3.8 (heal38 env)

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
PIPELINE_ROOT = HERE.parent.parent.parent  # scripts/build/<category>/ -> build/ -> scripts/ -> pipeline/
REPO_ROOT = str(PIPELINE_ROOT.parent.parent.parent)  # -> qcar_testbed_integration/
DEFAULT_CONF = HERE / "conf.json"


def _load_conf(path):
    """conf.json next to this script (shared by the three build_pretrain_*.py)
    is THE source of every default below -- nothing is hardcoded here. A CLI
    flag only overrides the matching key for that one run. Relative paths in
    it are resolved against the qcar_testbed_integration/ root (REPO_ROOT)."""
    if not os.path.exists(path):
        raise SystemExit("conf.json not found: %s" % path)
    with open(path) as f:
        return json.load(f)


def _path_from_conf(key):
    """conf.json path value; relative ones are resolved against REPO_ROOT."""
    v = _conf.get(key)
    return v if not v or os.path.isabs(str(v)) else os.path.join(REPO_ROOT, v)


_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--conf", default=str(DEFAULT_CONF))
_conf_path = _pre.parse_known_args()[0].conf
_conf = _load_conf(_conf_path)

# --- CLI flags, see qcar_dataset/pipeline/README.md's "Flag convention" ---
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument(
    "--conf", default=str(DEFAULT_CONF),
    help="Path to the conf.json to read defaults from (default: %s)" % DEFAULT_CONF)
_parser.add_argument(
    "--src-root", default=_path_from_conf("src_root"),
    help="Root of the 101-scenario collision corpus, external media (default: conf.json's src_root)")
_parser.add_argument(
    "--manifest-json", default=_path_from_conf("manifest_json"),
    help="pretrain_manifest.json path (default: conf.json's manifest_json)")
if __name__ == "__main__" and any(a in ("-h", "--help") for a in sys.argv[1:]):
    _parser.print_help()
    raise SystemExit(0)
_args, _ = _parser.parse_known_args()
_missing = [k for k in ("src_root", "manifest_json")
            if getattr(_args, k) is None]
if _missing:
    raise SystemExit("No value for %s: set it in %s or pass the flag."
                     % (", ".join("--" + k.replace("_", "-") for k in _missing), _conf_path))

SRC_ROOT = Path(_args.src_root)
OUT_PATH = Path(_args.manifest_json)
BLACK_THRESHOLD = 5.0  # mean pixel value below this = discard (warm-up frame)


def frame_ids(scenario_dir: Path) -> list[str]:
    return sorted(p.name for p in scenario_dir.iterdir()
                  if p.is_dir() and p.name.isdigit() and len(p.name) == 6)


def check_frame(frame_dir: Path) -> tuple[bool, str, bool]:
    """Returns (keep, reason_if_dropped, has_target)."""
    front = frame_dir / "front.png"
    if not front.exists():
        return False, "missing_front_png", False

    try:
        arr = np.asarray(Image.open(front))
    except Exception:
        return False, "unreadable_png", False
    if arr.mean() < BLACK_THRESHOLD:
        return False, "black_frame", False

    ego_f = frame_dir / "ego_vicon_pose.json"
    if not ego_f.exists():
        return False, "missing_ego_pose", False
    try:
        ego = json.loads(ego_f.read_text())
    except Exception:
        return False, "unreadable_ego_pose", False
    if ego.get("valid", 0.0) != 1.0:
        return False, "ego_pose_invalid", False

    target_f = frame_dir / "target_vicon_pose.json"
    has_target = False
    if target_f.exists():
        try:
            target = json.loads(target_f.read_text())
            has_target = target.get("valid", 0.0) == 1.0
        except Exception:
            has_target = False

    return True, "", has_target


def main() -> None:
    if not SRC_ROOT.exists():
        print(f"ERROR: source not mounted at {SRC_ROOT}", file=sys.stderr)
        sys.exit(1)

    scenarios = sorted(p for p in SRC_ROOT.iterdir() if p.is_dir())
    manifest: dict = {"source": str(SRC_ROOT), "black_threshold": BLACK_THRESHOLD,
                       "scenarios": {}}
    total_seen = total_kept = total_pos = 0
    drop_reasons: dict[str, int] = {}

    for i, scen_dir in enumerate(scenarios, 1):
        ids = frame_ids(scen_dir)
        kept_ids: list[str] = []
        kept_has_target: list[bool] = []
        n_pos = 0
        for fid in ids:
            keep, reason, has_target = check_frame(scen_dir / fid)
            total_seen += 1
            if keep:
                kept_ids.append(fid)
                kept_has_target.append(has_target)
                total_kept += 1
                if has_target:
                    n_pos += 1
                    total_pos += 1
            else:
                drop_reasons[reason] = drop_reasons.get(reason, 0) + 1

        manifest["scenarios"][scen_dir.name] = {
            "frames_seen": len(ids),
            "frames_kept": len(kept_ids),
            "positives_kept": n_pos,
            "kept_frame_ids": kept_ids,
            "kept_has_target": kept_has_target,
        }
        if i % 20 == 0 or i == len(scenarios):
            print(f"[{i}/{len(scenarios)}] scenarios processed, "
                  f"{total_kept}/{total_seen} frames kept so far", flush=True)

    manifest["totals"] = {
        "scenarios": len(scenarios),
        "frames_seen": total_seen,
        "frames_kept": total_kept,
        "positives_kept": total_pos,
        "negatives_kept": total_kept - total_pos,
        "drop_reasons": drop_reasons,
    }
    OUT_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {OUT_PATH}")
    print(f"Totals: {total_kept}/{total_seen} frames kept across "
          f"{len(scenarios)} scenarios "
          f"({total_pos} positive, {total_kept - total_pos} negative)")
    print(f"Drop reasons: {drop_reasons}")


if __name__ == "__main__":
    main()
