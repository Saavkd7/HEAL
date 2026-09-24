"""Pooling map for the QCar's three 160-degree cameras, in a x10-scaled world.

Two departures from the OPV2V rig the checkpoint was trained on:

**Three cameras, not four.** The capture has front, left and right; there is no
rear view. The cameras are combined by summation into the BEV grid and the
encoder runs per image with shared weights, so the count is a deployment
constant, not a learned property.

**The world is read at 1:10.** The model discretises depth from 2 to 50 m, while
the two QCars pass between 0.40 and 1.98 m -- every frame of the collision
scenario sits below its first depth bin, where no amount of training can place
an object. The QCar *is* a 1/10-scale car, so scaling the extrinsics by ten
returns the scene to the units the model was trained in: a car at 1.5 m is read
as one at 15 m, which is also what its apparent size already implies. The
geometry stays self-consistent because the camera-height-to-distance ratio is
unchanged (0.131/1.5 = 1.31/15).

The intrinsics are NOT scaled: they describe the lens, not the world.

    python3 build_qcar_pooling_map.py --output map.bin --scale 10 --fov 160

The historical default is ``--projection pinhole`` so old experiments remain
reproducible.  For the QCar's documented 160-degree horizontal fisheye,
``--projection equidistant`` unprojects image pixels into fisheye rays directly;
the static pooling table does not require the geometry to be expressible as a
3x3 pinhole intrinsic matrix.  It is still a nominal lens model, not a measured
checkerboard calibration.

``--native`` must equal the actual PAL capture size used to calibrate and run
the model.  Horizontal and vertical resize factors are handled independently;
the old scalar assumption was only correct for matching aspect ratios.

Measured extrinsics may be supplied as an NPZ with, for every camera name,
``CAMERA_R_camera_to_vehicle`` (OpenCV camera axes right/down/forward to QCar
axes forward/left/up) and ``CAMERA_t_vehicle_m`` (camera origin in the QCar
frame, physical metres). Translation, unlike intrinsics, is multiplied by the
world scale.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = str(HERE.parents[2])  # tools/ -> calibration_Matries/ -> calibration/ -> qcar_testbed_integration/
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


# conf is needed before the QuantV2X imports below (its location is a conf key).
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--conf", default=str(DEFAULT_CONF))
CONF = _load_conf(_pre.parse_known_args()[0].conf)

REPO = Path(CONF.path_of("pool_quantv2x_repo"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
for _p in CONF["pool_python_paths"]:  # e.g. where export_coop_stages.py lives
    sys.path.insert(0, _p if os.path.isabs(_p) else os.path.join(REPO_ROOT, _p))

from export_coop_stages import load_yaml                      # noqa: E402  (yaml fix included)
from opencood.tools.export_lss_pooling_map import make_frustum, geometry   # noqa: E402
from opencood.utils.camera_utils import gen_dx_bx             # noqa: E402

# Every view the QCar capture can provide, with its yaw in the vehicle frame.
# The encoder runs per image with shared weights, so which subset is used is a
# deployment choice, not a learned property -- `--cameras` selects it and the
# map's own length is what tells the CUDA agent how many there are.
ALL_CAMERA_YAW_DEG = {"front": 0.0, "left": 90.0, "right": -90.0}

# Populated from --cameras in main(); module-level so the geometry helpers and
# the measured-intrinsics/extrinsics loaders all agree on the same subset.
CAMERA_YAW_DEG = dict(ALL_CAMERA_YAW_DEG)


def select_cameras(names):
    """Restrict the module to a subset of views, preserving the given order."""
    unknown = [n for n in names if n not in ALL_CAMERA_YAW_DEG]
    if unknown:
        raise SystemExit("unknown camera(s) %s; known: %s"
                         % (unknown, list(ALL_CAMERA_YAW_DEG)))
    if not names:
        raise SystemExit("--cameras needs at least one view")
    global CAMERA_YAW_DEG
    CAMERA_YAW_DEG = {n: ALL_CAMERA_YAW_DEG[n] for n in names}
    return CAMERA_YAW_DEG


def focal_from_fov(projection, half_width, half_angle):
    if projection == "pinhole":
        return half_width / math.tan(half_angle)
    if projection == "equidistant":
        return half_width / half_angle
    if projection == "equisolid":
        return half_width / (2.0 * math.sin(half_angle / 2.0))
    if projection == "stereographic":
        return half_width / (2.0 * math.tan(half_angle / 2.0))
    if projection == "orthographic":
        return half_width / math.sin(half_angle)
    raise ValueError("unknown projection %s" % projection)


def build_calibration(fov_deg, native_wh, final_wh, scale, height_m, offset_m,
                      projection, extrinsics_path=None):
    width, height = native_wh
    focal = focal_from_fov(
        projection, width / 2.0, math.radians(fov_deg) / 2.0)
    intrinsics = np.array([[focal, 0, width / 2.0],
                           [0, focal, height / 2.0],
                           [0, 0, 1]], dtype=np.float32)
    rots, trans = [], []
    if extrinsics_path is not None:
        with np.load(str(extrinsics_path)) as measured:
            for name in CAMERA_YAW_DEG:
                rotation = np.asarray(
                    measured[name + "_R_camera_to_vehicle"], np.float32)
                translation = np.asarray(
                    measured[name + "_t_vehicle_m"], np.float32).reshape(3)
                if (rotation.shape != (3, 3) or
                        not np.isfinite(rotation).all() or
                        not np.isfinite(translation).all()):
                    raise ValueError("invalid extrinsics for %s" % name)
                if (not np.allclose(rotation.T @ rotation, np.eye(3),
                                    atol=1e-3) or
                        not np.isclose(np.linalg.det(rotation), 1.0,
                                       atol=1e-3)):
                    raise ValueError("%s rotation is not orthonormal" % name)
                rots.append(rotation)
                trans.append(translation * scale)
    else:
        for yaw in CAMERA_YAW_DEG.values():
            angle = math.radians(yaw)
            cos, sin = math.cos(angle), math.sin(angle)
            spin = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]],
                            dtype=np.float32)
            # a camera looks down its own +z; map that into the vehicle frame
            rots.append(spin @ np.array(
                [[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float32))
            trans.append([offset_m * scale * cos,
                          offset_m * scale * sin,
                          height_m * scale])
    resize_xy = (final_wh[0] / float(width),
                 final_wh[1] / float(height))
    post = np.diag([resize_xy[0], resize_xy[1], 1.0]).astype(np.float32)
    count = len(rots)
    return {"rots": np.stack(rots)[None],
            "trans": np.array(trans, dtype=np.float32)[None],
            "intrins": np.tile(intrinsics, (1, count, 1, 1)),
            "post_rots": np.tile(post, (1, count, 1, 1)),
            "post_trans": np.zeros((1, count, 3), dtype=np.float32)}, focal, resize_xy


def fisheye_geometry(calibration, frustum, native_wh, resize_xy, focal,
                     projection):
    """Unproject a radial fisheye image into the same camera-z depth geometry.

    OpenCOOD's LSS depth bins represent optical-axis depth: pinhole geometry
    forms ``[x/z*d, y/z*d, d]`` before applying camera extrinsics.  We preserve
    that convention and replace only the pixel-to-ray angle mapping.
    Rays at or behind 90 degrees have non-positive optical depth and are marked
    invalid; the model was not trained with a rear-facing half-space per view.
    """
    width, height = native_wh
    u = frustum[..., 0] / resize_xy[0] - width / 2.0
    v = frustum[..., 1] / resize_xy[1] - height / 2.0
    radius = torch.sqrt(u * u + v * v)
    scaled = radius / focal
    if projection == "equidistant":
        theta = scaled
    elif projection == "equisolid":
        theta = 2.0 * torch.asin(torch.clamp(scaled / 2.0, -1.0, 1.0))
    elif projection == "stereographic":
        theta = 2.0 * torch.atan(scaled / 2.0)
    elif projection == "orthographic":
        theta = torch.asin(torch.clamp(scaled, -1.0, 1.0))
    else:
        raise ValueError("fisheye_geometry requires a radial fisheye projection")

    safe_radius = torch.where(radius > 0, radius, torch.ones_like(radius))
    unit_u = torch.where(radius > 0, u / safe_radius, torch.zeros_like(u))
    unit_v = torch.where(radius > 0, v / safe_radius, torch.zeros_like(v))
    slope = torch.tan(theta)
    depth = frustum[..., 2]
    camera_points = torch.stack((
        slope * unit_u * depth,
        slope * unit_v * depth,
        depth,
    ), dim=-1)

    rots = torch.from_numpy(calibration["rots"])
    trans = torch.from_numpy(calibration["trans"])
    batch, cameras, _ = trans.shape
    expanded = camera_points.view(1, 1, *camera_points.shape).expand(
        batch, cameras, *camera_points.shape)
    points = rots.view(batch, cameras, 1, 1, 1, 3, 3).matmul(
        expanded.unsqueeze(-1)).squeeze(-1)
    points = points + trans.view(batch, cameras, 1, 1, 1, 3)
    ray_valid = (theta < math.radians(89.0)).view(
        1, 1, *theta.shape).expand(batch, cameras, *theta.shape)
    return points, ray_valid


def opencv_fisheye_geometry(calibration, frustum, native_wh, resize_xy,
                            intrinsics_path):
    """Unproject pixels with independently measured OpenCV fisheye K/D."""
    import cv2

    with np.load(str(intrinsics_path)) as measured:
        measured_size = tuple(int(value) for value in measured["image_size"])
        if measured_size != tuple(native_wh):
            raise ValueError("intrinsic image size %s does not match --native %s" %
                             (measured_size, native_wh))
        matrices = []
        distortions = []
        for name in CAMERA_YAW_DEG:
            matrices.append(np.asarray(measured[name + "_K"], np.float64))
            distortions.append(np.asarray(measured[name + "_D"], np.float64))

    # Pixel coordinates are the same at every depth, so unproject one HxW grid
    # per camera and broadcast its x/z,y/z slopes over the LSS depth bins.
    uv = frustum[0, ..., :2].numpy().astype(np.float64)
    uv[..., 0] /= resize_xy[0]
    uv[..., 1] /= resize_xy[1]
    depth = frustum[..., 2]
    camera_points, valid_rays = [], []
    for K, D in zip(matrices, distortions):
        normalized = cv2.fisheye.undistortPoints(
            uv.reshape(-1, 1, 2), K, D).reshape(*uv.shape)
        normalized = torch.from_numpy(normalized.astype(np.float32))
        camera_points.append(torch.stack((
            normalized[..., 0].unsqueeze(0) * depth,
            normalized[..., 1].unsqueeze(0) * depth,
            depth,
        ), dim=-1))
        radius = torch.sqrt((normalized * normalized).sum(dim=-1))
        valid_rays.append(torch.isfinite(radius) &
                          (torch.atan(radius) < math.radians(89.0)))

    camera_points = torch.stack(camera_points, dim=0).unsqueeze(0)
    rots = torch.from_numpy(calibration["rots"])
    trans = torch.from_numpy(calibration["trans"])
    batch, cameras, _ = trans.shape
    points = rots.view(batch, cameras, 1, 1, 1, 3, 3).matmul(
        camera_points.unsqueeze(-1)).squeeze(-1)
    points = points + trans.view(batch, cameras, 1, 1, 1, 3)
    ray_valid = torch.stack(valid_rays, dim=0)
    ray_valid = ray_valid.view(1, cameras, 1, *ray_valid.shape[-2:]).expand(
        batch, cameras, depth.shape[0], *ray_valid.shape[-2:])
    return points, ray_valid


def main():
    conf = CONF

    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", default=str(DEFAULT_CONF),
                     help="Path to the conf.json to read defaults from (default: %s)" % DEFAULT_CONF)
    ap.add_argument("--config", default=conf.path_of("pool_config"))
    ap.add_argument("--output", type=Path, default=conf.path_of("pool_output"))
    ap.add_argument("--cameras", nargs="+", default=conf["pool_cameras"],
                    metavar="NAME",
                    help=("which views to build the map for, in order (default: "
                          "conf.json's pool_cameras). Use e.g. "
                          "`--cameras front` for a single-camera deployment: "
                          "the encoder is per-image with shared weights, so no "
                          "retraining is involved, and coop_agent reads the "
                          "camera count back out of the map's own length."))
    ap.add_argument("--fov", type=float, default=conf["pool_fov"])
    ap.add_argument("--projection", choices=("pinhole", "equidistant", "equisolid",
                                               "stereographic", "orthographic",
                                               "opencv_fisheye"),
                    default=conf["pool_projection"])
    ap.add_argument("--intrinsics", type=Path, default=conf.path_of("pool_intrinsics"),
                    help="three-camera K/D .npz required by opencv_fisheye")
    ap.add_argument("--extrinsics", type=Path, default=conf.path_of("pool_extrinsics"),
                    help=("optional .npz with CAMERA_R_camera_to_vehicle and "
                          "CAMERA_t_vehicle_m for front/left/right"))
    ap.add_argument("--scale", type=float, default=conf["pool_scale"],
                     help="world scale; 1 disables the rescaling")
    ap.add_argument("--height", type=float, default=conf["pool_height"],
                     help="camera height in real metres")
    ap.add_argument("--offset", type=float, default=conf["pool_offset"],
                     help="camera offset from centre, real metres")
    ap.add_argument("--native", default=conf["pool_native"])
    ap.add_argument("--ignore-z-bound", action="store_true", default=conf["pool_ignore_z_bound"],
                    help="reproduce the initial diagnostic map's incorrect x/y-only filter")
    args = ap.parse_args()
    select_cameras(args.cameras)

    hypes = load_yaml(args.config, None)
    encoder = hypes["model"]["args"]["m2"]["encoder_args"]
    grid_conf, aug, downsample = encoder["grid_conf"], encoder["data_aug_conf"], encoder["img_downsample"]
    native = tuple(int(v) for v in args.native.split("x"))
    final = (aug["final_dim"][1], aug["final_dim"][0])

    if args.projection == "opencv_fisheye" and args.intrinsics is None:
        raise SystemExit("--intrinsics is required for --projection opencv_fisheye")
    nominal_projection = ("equidistant" if args.projection == "opencv_fisheye"
                          else args.projection)
    calibration, focal, resize_xy = build_calibration(
        args.fov, native, final, args.scale, args.height, args.offset,
        nominal_projection, args.extrinsics)
    cameras = calibration["trans"].shape[1]

    frustum = make_frustum(grid_conf, aug, downsample)
    if args.projection == "pinhole":
        points = geometry(calibration, frustum)                # (1, N, D, H, W, 3)
        ray_valid = torch.ones(points.shape[:-1], dtype=torch.bool)
    elif args.projection == "opencv_fisheye":
        points, ray_valid = opencv_fisheye_geometry(
            calibration, frustum, native, resize_xy, args.intrinsics)
    else:
        points, ray_valid = fisheye_geometry(
            calibration, frustum, native, resize_xy, focal, args.projection)

    dx, bx, nx = gen_dx_bx(grid_conf["xbound"], grid_conf["ybound"], grid_conf["zbound"])
    cells = ((points - (bx - dx / 2.0)) / dx).long()
    inside = (ray_valid &
              (cells[..., 0] >= 0) & (cells[..., 0] < nx[0]) &
              (cells[..., 1] >= 0) & (cells[..., 1] < nx[1]))
    if not args.ignore_z_bound:
        inside = (inside & (cells[..., 2] >= 0) & (cells[..., 2] < nx[2]))
    index = torch.where(inside, cells[..., 1] * int(nx[0]) + cells[..., 0],
                        torch.full_like(cells[..., 0], -1)).to(torch.int32)

    array = index.numpy().astype(np.int32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    array.tofile(args.output)
    valid = int((array >= 0).sum())
    meta = {"cameras": cameras, "camera_names": list(CAMERA_YAW_DEG),
            "fov_deg": args.fov, "focal_px_native": focal,
            "native_image_wh": list(native),
            "model_image_wh": list(final),
            "resize_xy": list(resize_xy),
            "projection": args.projection,
            "intrinsics": str(args.intrinsics) if args.intrinsics else None,
            "extrinsics": str(args.extrinsics) if args.extrinsics else None,
            "world_scale": args.scale, "camera_height_m": args.height * args.scale,
            "camera_offset_m": args.offset * args.scale,
            "z_bound_applied": not args.ignore_z_bound,
            "shape": list(array.shape), "dtype": "int32",
            "valid_points": valid, "total_points": int(array.size),
            "bev_shape": [int(nx[1]), int(nx[0])],
            "warning": (
                "measured fisheye intrinsics and supplied camera-to-vehicle extrinsics"
                if args.projection == "opencv_fisheye" and args.extrinsics else
                "measured fisheye intrinsics; extrinsics still require validation"
                if args.projection == "opencv_fisheye" else
                "nominal lens model; diagnostic only until measured calibration"),
            "layout": "N,D,H,W; value=y*bev_width+x; -1 means outside BEV"}
    Path(str(args.output) + ".json").write_text(json.dumps(meta, indent=2) + "\n")
    print("cameras %d (%s)  %s fov %.0f  focal %.1f px  world scale x%.0f" %
          (cameras, ", ".join(CAMERA_YAW_DEG), args.projection,
           args.fov, focal, args.scale))
    print("valid points %d / %d = %.2f%%" % (valid, array.size, 100.0 * valid / array.size))
    print("wrote %s and its .json" % args.output.name)


if __name__ == "__main__":
    main()
