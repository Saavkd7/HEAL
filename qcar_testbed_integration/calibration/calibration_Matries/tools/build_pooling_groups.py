"""Turn a Lift-Splat pooling map into a cell-grouped plan, once, offline.

The pooling map is a static file: the same table serves every frame, every
scene and both cars. The original Lift-Splat implementation sorts points per
sample because PyTorch recomputes the geometry each time; we have no such
constraint, so the sorting, compaction and segmentation are build-time work.

Emits, beside `<map>.bin`, a `<map>.bin.groups` containing

    int32 n_cells
    int32 n_points
    int32 cells[n_cells]        BEV cell id owned by each block
    int32 offsets[n_cells + 1]  where each cell's points start and end
    int32 points[n_points]      original point indices, grouped by cell

which lets the kernel accumulate a whole cell in a register and store it once,
instead of issuing one global atomic per point per channel.
"""
import argparse
import json
import os
import struct
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))  # tools/ -> calibration_Matries/ -> calibration/ -> qcar_testbed_integration/
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



def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--conf", default=DEFAULT_CONF)
    conf_path = pre.parse_known_args()[0].conf
    conf = _load_conf(conf_path)

    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", default=DEFAULT_CONF,
                     help="Path to the conf.json to read defaults from (default: %s)" % DEFAULT_CONF)
    ap.add_argument("map", type=Path, nargs="?",
                     default=conf.path_of("pooling_groups_map"),
                     help="pooling_index .bin")
    ap.add_argument("--output", type=Path,
                     default=conf.path_of("pooling_groups_output"))
    args = ap.parse_args()
    if args.map is None:
        ap.error("map not set: pass it positionally, or set \"pooling_groups_map\" in %s" % conf_path)

    index = np.fromfile(args.map, dtype=np.int32)
    valid = np.nonzero(index >= 0)[0].astype(np.int32)
    cells_of_valid = index[valid]

    order = np.argsort(cells_of_valid, kind="stable")     # group points by cell
    points = valid[order]
    sorted_cells = cells_of_valid[order]
    cells, starts = np.unique(sorted_cells, return_index=True)
    offsets = np.append(starts, len(points)).astype(np.int32)

    out = args.output or Path(str(args.map) + ".groups")
    with out.open("wb") as handle:
        handle.write(struct.pack("<ii", len(cells), len(points)))
        cells.astype(np.int32).tofile(handle)
        offsets.tofile(handle)
        points.astype(np.int32).tofile(handle)

    counts = np.diff(offsets)
    print("map            : %s  (%d entries)" % (args.map.name, index.size))
    print("valid points   : %d  (%.2f%%)" % (len(points), 100.0 * len(points) / index.size))
    print("occupied cells : %d of %d" % (len(cells), 256 * 256))
    print("points per cell: mean %.1f  median %d  max %d" %
          (counts.mean(), int(np.median(counts)), counts.max()))
    print("atomics before : %d" % (len(points) * 128))
    print("stores after   : %d  (%.1fx fewer)" %
          (len(cells) * 128, len(points) / float(len(cells))))
    print("wrote          : %s  (%.1f MB)" % (out.name, out.stat().st_size / 2**20))


if __name__ == "__main__":
    main()
