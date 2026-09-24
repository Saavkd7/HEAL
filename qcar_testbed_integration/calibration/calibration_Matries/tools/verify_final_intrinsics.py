"""Fit and verify the qcar-52775 front fisheye intrinsics."""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
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


_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--conf", default=DEFAULT_CONF)
_conf = _load_conf(_pre.parse_known_args()[0].conf)

_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument("--conf", default=DEFAULT_CONF,
                     help="conf.json to read defaults from (default: %s)" % DEFAULT_CONF)
_parser.add_argument("--views-dir", default=_conf.path_of("verify_views_dir"),
                     help="Earlier accepted chessboard views (default: conf.json's verify_views_dir)")
_parser.add_argument("--initial", default=_conf.path_of("verify_initial"),
                     help="Initial intrinsics .npz (default: conf.json's verify_initial)")
_parser.add_argument("--new-capture", default=_conf.path_of("verify_new_capture"),
                     help="New capture folder of views (default: conf.json's verify_new_capture)")
_parser.add_argument("--report", default=_conf.path_of("verify_report"),
                     help="Where to write the JSON report (default: conf.json's verify_report)")
_parser.add_argument("--output", default=_conf.path_of("verify_output"),
                     help="Where to write the verified .npz (default: conf.json's verify_output)")
_args, _ = _parser.parse_known_args()

CAR_NAME = _conf["verify_car_name"]
PATTERN = tuple(_conf["verify_pattern"])
SQUARE_M = _conf["verify_square_m"]
IMAGE_SIZE = tuple(_conf["verify_image_size"])
VIEWS_DIR = Path(_args.views_dir)
INITIAL = Path(_args.initial)
NEW_CAPTURE = Path(_args.new_capture)
REPORT = Path(_args.report)
OUTPUT = Path(_args.output)


def detect(path_text):
    cv2.setNumThreads(1)
    path = Path(path_text)
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None or gray.shape[::-1] != IMAGE_SIZE:
        return path_text, None
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
             cv2.CALIB_CB_NORMALIZE_IMAGE |
             cv2.CALIB_CB_FAST_CHECK)
    ok, corners = cv2.findChessboardCorners(gray, PATTERN, flags)
    if not ok:
        return path_text, None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                40, 1e-5)
    corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
    return path_text, corners.reshape(1, -1, 2).astype(np.float64)


OBJECT = np.zeros((1, PATTERN[0] * PATTERN[1], 3), np.float64)
OBJECT[0, :, :2] = (
    np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQUARE_M)


def errors_for(points, K, D, rvecs, tvecs):
    errors = []
    for observed, rvec, tvec in zip(points, rvecs, tvecs):
        projected, _ = cv2.fisheye.projectPoints(OBJECT, rvec, tvec, K, D)
        delta = projected.reshape(-1, 2) - observed.reshape(-1, 2)
        errors.append(float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))))
    return np.asarray(errors)


def fit(points, initial_K, initial_D, fix_high_order):
    flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC |
             cv2.fisheye.CALIB_FIX_SKEW |
             cv2.fisheye.CALIB_USE_INTRINSIC_GUESS)
    if fix_high_order:
        flags |= cv2.fisheye.CALIB_FIX_K3 | cv2.fisheye.CALIB_FIX_K4
    K = initial_K.copy()
    D = initial_D.copy()
    if fix_high_order:
        D[2:] = 0.0
    result = cv2.fisheye.calibrate(
        [OBJECT.copy() for _ in points], points, IMAGE_SIZE, K, D,
        None, None, flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                  150, 1e-9))
    rms, K, D, rvecs, tvecs = result
    errors = errors_for(points, K, D, rvecs, tvecs)
    return float(rms), K, D, errors


def fold_angle(D):
    k1, k2, k3, k4 = D.reshape(4)
    theta = np.linspace(0.0, np.radians(90.0), 10000)
    derivative = (1.0 + 3*k1*theta**2 + 5*k2*theta**4 +
                  7*k3*theta**6 + 9*k4*theta**8)
    bad = np.flatnonzero(derivative <= 0.0)
    return float(np.degrees(theta[bad[0]])) if len(bad) else 90.0


def fit_with_outlier_rejection(points, K0, D0, fix_high_order):
    rms, K, D, errors = fit(points, K0, D0, fix_high_order)
    median = float(np.median(errors))
    mad = float(np.median(np.abs(errors - median)))
    cutoff = max(1.25, median + 3.0 * max(mad, 0.1))
    keep = np.flatnonzero(errors <= cutoff)
    if 40 <= len(keep) < len(points):
        kept = [points[index] for index in keep]
        rms, K, D, errors = fit(kept, K, D, fix_high_order)
    else:
        kept = points
    return kept, rms, K, D, errors, cutoff


def candidate_report(name, kept, rms, K, D, errors, cutoff):
    corners = np.concatenate([item.reshape(-1, 2) for item in kept])
    span = corners.max(axis=0) - corners.min(axis=0)
    coverage = [float(span[0] / IMAGE_SIZE[0]),
                float(span[1] / IMAGE_SIZE[1])]
    return {
        "name": name,
        "views_used": len(kept),
        "rms_px": rms,
        "median_view_error_px": float(np.median(errors)),
        "p95_view_error_px": float(np.percentile(errors, 95)),
        "max_view_error_px": float(errors.max()),
        "outlier_cutoff_px": cutoff,
        "coverage_fraction_xy": coverage,
        "fold_angle_deg": fold_angle(D),
        "K": K.tolist(),
        "D": D.reshape(4).tolist(),
    }


def point_holdout_error(observed, K, D):
    normalized = cv2.fisheye.undistortPoints(observed, K, D)
    board_xy = OBJECT.reshape(-1, 3)
    selector = np.arange(len(board_xy)) % 2 == 0
    ok, rvec, tvec = cv2.solvePnP(
        board_xy[selector], normalized.reshape(-1, 2)[selector],
        np.eye(3), np.zeros(4), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    projected, _ = cv2.fisheye.projectPoints(OBJECT, rvec, tvec, K, D)
    delta = (projected.reshape(-1, 2)[~selector] -
             observed.reshape(-1, 2)[~selector])
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def main():
    paths = sorted(VIEWS_DIR.glob("*.png"))
    paths += sorted(NEW_CAPTURE.glob("*.png"))
    with ProcessPoolExecutor(max_workers=8) as pool:
        detected = list(pool.map(detect, map(str, paths), chunksize=8))
    observations = [(Path(path), corners) for path, corners in detected
                    if corners is not None]
    points = [corners for _, corners in observations]
    with np.load(str(INITIAL)) as initial:
        K0 = np.asarray(initial["front_K"], np.float64)
        D0 = np.asarray(initial["front_D"], np.float64).reshape(4, 1)

    candidates = []
    fits = []
    for name, fixed in (("full_k1_k4", False), ("fixed_k3_k4", True)):
        kept, rms, K, D, errors, cutoff = fit_with_outlier_rejection(
            points, K0, D0, fixed)
        report = candidate_report(name, kept, rms, K, D, errors, cutoff)
        candidates.append(report)
        fits.append((kept, rms, K, D, errors, fixed))

    valid_indices = [index for index, item in enumerate(candidates)
                     if item["fold_angle_deg"] >= 80.0]
    if not valid_indices:
        chosen_index = int(np.argmin([item["rms_px"] for item in candidates]))
    else:
        chosen_index = min(valid_indices,
                           key=lambda index: candidates[index]["rms_px"])
    selected = candidates[chosen_index]
    kept, rms, K, D, errors, fixed = fits[chosen_index]

    # Independent point holdout: pose uses half the checker intersections and
    # error is evaluated on the other half.
    holdout_errors = [point_holdout_error(item, K, D) for item in kept]
    holdout_errors = np.asarray([item for item in holdout_errors
                                 if item is not None])
    selected["point_holdout_median_px"] = float(np.median(holdout_errors))
    selected["point_holdout_p95_px"] = float(np.percentile(holdout_errors, 95))
    gates = {
        "views_at_least_40": len(kept) >= 40,
        "coverage_x_at_least_70pct": selected["coverage_fraction_xy"][0] >= 0.70,
        "coverage_y_at_least_70pct": selected["coverage_fraction_xy"][1] >= 0.70,
        "rms_below_0_5px": rms < 0.5,
        "monotonic_to_80deg": selected["fold_angle_deg"] >= 80.0,
        "point_holdout_p95_below_2px": selected["point_holdout_p95_px"] < 2.0,
    }
    passed = bool(all(gates.values()))
    report = {
        "car": CAR_NAME,
        "camera": "front",
        "images_examined": len(paths),
        "views_detected": len(points),
        "selected_model": selected["name"],
        "candidates": candidates,
        "gates": gates,
        "all_gates_pass": passed,
        "safe_for_heal_geometry_trial": passed,
        "note": "Front camera only; three-camera HEAL still needs left/right calibration.",
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    np.savez(str(OUTPUT), front_K=K, front_D=D.reshape(4),
             image_size=np.asarray(IMAGE_SIZE, dtype=np.int32))
    print(json.dumps({
        "views": len(kept), "model": selected["name"], "rms_px": rms,
        "coverage": selected["coverage_fraction_xy"],
        "fold_angle_deg": selected["fold_angle_deg"],
        "holdout_p95_px": selected["point_holdout_p95_px"],
        "passed": passed, "K": K.tolist(), "D": D.reshape(4).tolist(),
    }, indent=2))


if __name__ == "__main__":
    main()
