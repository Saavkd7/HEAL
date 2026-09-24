"""Build a leakage-reduced development split for the 113-frame QCar run.

Renamed 2026-09-18: `OnlyFront`/`OnlyFront_dev` -> `SmokeTestFront`/
`SmokeTestFront_dev`. "OnlyFront" stopped meaning anything once `CoopFront`
became single-camera too; this dataset's real, current use is a fast
smoke-test fixture (69MB total, one real completed training run,
`opencood/logs/HeterBaseline_opv2v_camera_attfuse_2026_09_12_18_27_33/`) for
checking a loader/patch change didn't break anything, without waiting on the
full `CoopFront` cycle -- see `CONVERSION_NOTES.md`.

The original chronological split placed every usable cooperative positive in
train and none in validation/test.  This builder preserves that dataset and
creates ``SmokeTestFront_dev`` with temporal-block validation and five-frame
(approximately 0.5 s) purge bands.  A hard-linked ``final_train`` view contains
all 113 frames for the final fit after development choices are frozen.

No local test split is created: an eight-frame tail from the same continuous
trajectory is not an independent test.  Final evaluation needs a new run.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
REPO_ROOT = str(HERE.parents[5])  # tools/ -> build/ -> scripts/ -> pipeline/ -> qcar_dataset/ -> data/ -> qcar_testbed_integration/
DEFAULT_CONF = HERE / "conf.json"


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
_pre.add_argument("--conf", default=str(DEFAULT_CONF))
_conf = _load_conf(_pre.parse_known_args()[0].conf)

_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument("--conf", default=str(DEFAULT_CONF),
                     help="conf.json to read defaults from (default: %s)" % DEFAULT_CONF)
_parser.add_argument("--source", default=_conf.path_of("smoketest_source"),
                     help="SmokeTestFront/ to re-split (default: conf.json's smoketest_source)")
_parser.add_argument("--destination", default=_conf.path_of("smoketest_destination"),
                     help="Where to write the dev split (default: conf.json's smoketest_destination)")
_args, _ = _parser.parse_known_args()

SOURCE = Path(_args.source)
DESTINATION = Path(_args.destination)
BUILDING = DESTINATION.with_name(DESTINATION.name + ".building")
SCENARIO = _conf["smoketest_scenario"]
CAV_IDS = tuple(_conf["smoketest_cav_ids"])
SOURCE_SPLITS = ("train", "validate", "test")

# Positive cooperative GT is present on frames 0..70 and absent on 71..112,
# as measured through the real HEAL DataLoader before defining this split.
POSITIVE_IDS = set(range(0, 71))
ALL_IDS = list(range(113))
VALIDATE_IDS = list(range(51, 61)) + list(range(93, 103))
PURGED_IDS = list(range(46, 51)) + list(range(61, 66)) + \
    list(range(88, 93)) + list(range(103, 108))
TRAIN_IDS = sorted(set(ALL_IDS) - set(VALIDATE_IDS) - set(PURGED_IDS))


def hardlink_or_copy(source: Path, destination: Path) -> str:
    """Hard-link immutable image data, falling back to a metadata copy."""
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def source_index() -> dict[int, str]:
    """Map each frame ID to its original split without changing the source."""
    result = {}
    for split in SOURCE_SPLITS:
        cav_dir = SOURCE / split / SCENARIO / CAV_IDS[0]
        for yaml_path in sorted(cav_dir.glob("*.yaml")):
            frame_id = int(yaml_path.stem)
            if frame_id in result:
                raise RuntimeError("Duplicate source frame %06d" % frame_id)
            result[frame_id] = split
    expected = set(ALL_IDS)
    if set(result) != expected:
        raise RuntimeError(
            "Source frame IDs differ: missing=%s extra=%s" %
            (sorted(expected - set(result)), sorted(set(result) - expected))
        )
    return result


def write_view(
    name: str,
    frame_ids: list[int],
    source_splits: dict[int, str],
) -> tuple[int, int]:
    """Write one dataset view and return hard-link/copy counts."""
    hardlinks = 0
    copies = 0
    for frame_id in frame_ids:
        frame = "%06d" % frame_id
        original_split = source_splits[frame_id]
        for cav_id in CAV_IDS:
            source_dir = SOURCE / original_split / SCENARIO / cav_id
            destination_dir = BUILDING / name / SCENARIO / cav_id
            destination_dir.mkdir(parents=True, exist_ok=True)

            for suffix in ("_camera0.png", "_bev_visibility.png"):
                source_file = source_dir / (frame + suffix)
                destination_file = destination_dir / source_file.name
                if not source_file.is_file():
                    raise FileNotFoundError(source_file)
                mode = hardlink_or_copy(source_file, destination_file)
                hardlinks += mode == "hardlink"
                copies += mode == "copy"

            source_yaml = source_dir / (frame + ".yaml")
            with source_yaml.open("r", encoding="utf-8") as stream:
                params = yaml.safe_load(stream)
            provenance = params.setdefault("qcar_provenance", {})
            provenance["original_split"] = original_split
            provenance["split"] = name
            provenance["development_view"] = name
            provenance["world_scale"] = (
                "x10 only on horizontal xy and vehicle length/width; z, "
                "height, rotations, and camera intrinsics are not scaled"
            )
            provenance["development_split_policy"] = (
                "temporal blocks with a five-frame purge around validation; "
                "no local test"
            )
            with (destination_dir / source_yaml.name).open(
                "w", encoding="utf-8"
            ) as stream:
                yaml.safe_dump(params, stream, sort_keys=False)
    return hardlinks, copies


def main() -> None:
    if DESTINATION.exists() or BUILDING.exists():
        raise FileExistsError(
            "Refusing to overwrite %s or %s" % (DESTINATION, BUILDING)
        )
    if not SOURCE.is_dir():
        raise FileNotFoundError(SOURCE)

    if set(TRAIN_IDS) & set(VALIDATE_IDS):
        raise RuntimeError("Train/validation overlap")
    if set(PURGED_IDS) & (set(TRAIN_IDS) | set(VALIDATE_IDS)):
        raise RuntimeError("Purged-frame overlap")
    if set(TRAIN_IDS) | set(VALIDATE_IDS) | set(PURGED_IDS) != set(ALL_IDS):
        raise RuntimeError("Development partitions do not cover all frames")
    minimum_separation = min(
        abs(train_id - validation_id)
        for train_id in TRAIN_IDS
        for validation_id in VALIDATE_IDS
    )
    if minimum_separation != 6:
        raise RuntimeError("Unexpected train/validation separation")

    sources = source_index()
    totals = {"hardlinks": 0, "copies": 0}
    for view, frame_ids in (
        ("train", TRAIN_IDS),
        ("validate", VALIDATE_IDS),
        ("final_train", ALL_IDS),
    ):
        hardlinks, copies = write_view(view, frame_ids, sources)
        totals["hardlinks"] += hardlinks
        totals["copies"] += copies

    manifest = {
        "built": "2026-09-12",
        "source_preserved": str(SOURCE),
        "scenario": SCENARIO,
        "purpose": "development and training-loop validation; not final AP",
        "camera_count_per_agent": 1,
        "world_scale": (
            "x10 on horizontal xy and vehicle length/width only; z, height, "
            "rotations, and intrinsics remain physical"
        ),
        "observed_cooperative_gt": {
            "positive_frame_ids": [0, 70],
            "positive_count": len(POSITIVE_IDS),
            "negative_frame_ids": [71, 112],
            "negative_count": len(ALL_IDS) - len(POSITIVE_IDS),
            "definition": "object_bbx_mask.sum()>0 through the real HEAL DataLoader",
        },
        "views": {
            "train": {
                "frame_ids": TRAIN_IDS,
                "count": len(TRAIN_IDS),
                "positive": len(set(TRAIN_IDS) & POSITIVE_IDS),
                "negative": len(set(TRAIN_IDS) - POSITIVE_IDS),
            },
            "validate": {
                "frame_ids": VALIDATE_IDS,
                "count": len(VALIDATE_IDS),
                "positive": len(set(VALIDATE_IDS) & POSITIVE_IDS),
                "negative": len(set(VALIDATE_IDS) - POSITIVE_IDS),
            },
            "purged_from_development": {
                "frame_ids": PURGED_IDS,
                "count": len(PURGED_IDS),
                "reason": (
                    "five-frame (~0.5 s at ~10 Hz) buffer on each temporal "
                    "validation boundary"
                ),
            },
            "final_train": {
                "frame_ids": ALL_IDS,
                "count": len(ALL_IDS),
                "use": (
                    "fit once after hyperparameters are frozen; evaluate on "
                    "a new independent trajectory"
                ),
            },
            "test": None,
        },
        "minimum_train_validate_frame_distance": minimum_separation,
        "split_policy": (
            "purged temporal-block development validation; no random adjacent "
            "frame split"
        ),
        "test_policy": (
            "No same-trajectory test. Collect a new QCar trajectory after "
            "development choices are frozen."
        ),
        "storage": totals,
    }
    with (BUILDING / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    os.replace(BUILDING, DESTINATION)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
