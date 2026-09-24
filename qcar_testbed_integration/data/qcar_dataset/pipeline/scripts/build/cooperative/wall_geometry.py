"""Real wall-segment occlusion geometry for the physical QCar lab.

Wall positions come from shared/walls.json (real, unscaled Vicon-frame
metres -- the same frame as STATIC_TARGETS x/y and raw agent poses in
build_coop_train_val_dataset.py, NOT the x10-scaled OPV2V coordinates written into
the yaml). Measured from media/Walls_Layout+Scenario1.jpeg, user-confirmed
2026-09-22. Add more walls by appending an entry to that JSON file -- this
module re-reads it, no code change needed.

Replaces build_coop_train_val_dataset.py's static, hand-confirmed WALL_BLOCKED fact
(a fixed set of (agent_id, target_id) pairs, always blocked regardless of
where the agent actually is) with a real PER-FRAME check: does the straight
line between the agent's real position and the target's real position cross
any wall segment. This is strictly more informative than WALL_BLOCKED
because agents move -- a pair can be blocked at one point in a trajectory
and clear at another, which a fixed pair-level fact can never capture.

    python wall_geometry.py    # smoke-test: prints the loaded walls
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def load_walls(path):
    """Returns a list of (a, b, id) tuples, a/b = (x, y) real-metre
    endpoints, from the given walls.json. The path has no default here --
    callers pass conf.json's walls_json (build_coop_train_val_dataset.WALLS)."""
    with open(path) as f:
        doc = json.load(f)
    return [(tuple(w["a"]), tuple(w["b"]), w.get("id", "")) for w in doc["walls"]]


def _ccw(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(p1, p2, p3, p4):
    """Standard CCW-orientation segment-intersection test: does segment
    p1-p2 cross segment p3-p4."""
    return _ccw(p1, p3, p4) != _ccw(p2, p3, p4) and _ccw(p1, p2, p3) != _ccw(p1, p2, p4)


def blocked_by_wall(p_from, p_to, walls):
    """True if the straight line from p_from to p_to (real-metre x,y
    tuples) crosses ANY wall segment in `walls` (as returned by
    load_walls())."""
    for a, b, _id in walls:
        if segments_intersect(p_from, p_to, a, b):
            return True
    return False


def which_wall_blocks(p_from, p_to, walls):
    """Like blocked_by_wall, but returns the blocking wall's id (or None)
    -- useful for diagnostics/attribution."""
    for a, b, wid in walls:
        if segments_intersect(p_from, p_to, a, b):
            return wid
    return None


if __name__ == "__main__":
    import sys
    # walls_json from cooperative/conf.json (relative to qcar_testbed_integration/),
    # or a path passed as the only argument.
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))))
        with open(os.path.join(HERE, "conf.json")) as f:
            path = json.load(f)["walls_json"]
        path = path if os.path.isabs(path) else os.path.join(repo_root, path)
    walls = load_walls(path)
    print("Loaded %d wall(s) from %s:" % (len(walls), path))
    for a, b, wid in walls:
        print("  %-8s %s -- %s" % (wid, a, b))
