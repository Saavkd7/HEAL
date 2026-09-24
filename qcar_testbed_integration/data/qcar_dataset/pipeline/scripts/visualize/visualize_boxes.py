"""Draw the 3D box we generate (from Vicon position + known QCar size) onto
its own undistorted image, so distortion/placement can be checked by eye --
especially far/edge-of-frame cases, where fisheye undistortion is weakest.

Projection: box corners (world/vehicle-body frame, ego at origin) ->
camera frame via the manual's R_camera_to_vehicle (inverted) and the yaml's
camera0.cords position -> pinhole projection with the verified K (the image
is already undistorted to be pinhole-consistent with this same K).

Usage:
    python3 visualize_boxes.py <frame_dir> [<frame_dir> ...] --out <dir>

Each <frame_dir> is a PretrainFront/<split>/<scenario>/1 folder; pass
specific frame ids via --frames 000005,000030 or it samples a spread
automatically.
"""
import argparse
import glob
import os

import cv2
import numpy as np
import yaml

# Manual value (Quanser QCar hardware manual p.8), camera-to-vehicle rotation.
R_CB = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)
SCALE = 10.0  # the label's location/extent x,y are x10 model-space; camera
              # optics only know real metres, so undo it just for this check.


def box_corners_world(location, center, extent, angle_deg_y):
    # location/extent x,y are x10 model-space; z/height already real.
    cx = location[0] / SCALE + center[0]
    cy = location[1] / SCALE + center[1]
    cz = location[2] + center[2]
    l, w, h = extent[0] / SCALE, extent[1] / SCALE, extent[2]  # half-extents
    yaw = np.radians(angle_deg_y)
    c, s = np.cos(yaw), np.sin(yaw)
    local = np.array([
        [l, w, h], [l, -w, h], [-l, -w, h], [-l, w, h],
        [l, w, -h], [l, -w, -h], [-l, -w, -h], [-l, w, -h],
    ])
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    world = local @ rot.T + np.array([cx, cy, cz])
    return world


def project(points_world, cam_cords, K):
    t_cb = np.asarray(cam_cords[:3], dtype=np.float64)
    t_cb = np.array([t_cb[0] / SCALE, t_cb[1] / SCALE, t_cb[2]])  # x,y were x10 too
    rel = points_world - t_cb
    cam = rel @ R_CB  # R_CB^T @ rel, applied row-wise as rel @ R_CB
    depth = cam[:, 2]
    valid = depth > 0.05
    uvw = cam @ K.T
    uv = uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-6, None)
    return uv, valid, depth


EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),
         (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]


def draw_frame(frame_dir, out_dir):
    frame_id = os.path.basename(sorted(glob.glob(frame_dir + "/*_camera0.png"))[0]).split("_")[0]
    img_path = os.path.join(frame_dir, f"{frame_id}_camera0.png")
    yaml_path = os.path.join(frame_dir, f"{frame_id}.yaml")
    if not os.path.exists(img_path) or not os.path.exists(yaml_path):
        return None
    params = yaml.safe_load(open(yaml_path))
    if not params.get("vehicles"):
        return None  # negative frame, nothing to draw

    img = cv2.imread(img_path)
    K = np.array(params["camera0"]["intrinsic"], dtype=np.float64)
    cam_cords = params["camera0"]["cords"]

    veh = list(params["vehicles"].values())[0]
    corners = box_corners_world(veh["location"], veh["center"], veh["extent"],
                                 veh["angle"][1])
    uv, valid, depth = project(corners, cam_cords, K)

    for i, j in EDGES:
        if not (valid[i] and valid[j]):
            continue
        p1 = tuple(np.round(uv[i]).astype(int))
        p2 = tuple(np.round(uv[j]).astype(int))
        cv2.line(img, p1, p2, (0, 255, 0), 2)

    dist_m = float(np.linalg.norm(np.mean(corners, axis=0)[:2]))
    label = f"{frame_id}  dist={dist_m:.2f}m  depth_range=[{depth.min():.1f},{depth.max():.1f}]"
    cv2.putText(img, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    out_path = os.path.join(out_dir, f"{os.path.basename(frame_dir.rstrip('/'))}_{frame_id}.png")
    cv2.imwrite(out_path, img)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario_dirs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", default=None, help="comma-separated frame ids; default: spread")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for scen_dir in args.scenario_dirs:
        yaml_files = sorted(glob.glob(os.path.join(scen_dir, "*.yaml")))
        if not yaml_files:
            continue
        if args.frames:
            ids = args.frames.split(",")
        else:
            n = len(yaml_files)
            picks = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))
            ids = [os.path.splitext(os.path.basename(yaml_files[p]))[0] for p in picks]
        for fid in ids:
            fdir_alias = scen_dir  # reuse; draw_frame finds any *_camera0.png, so scope one id
            single_glob = glob.glob(os.path.join(scen_dir, f"{fid}_camera0.png"))
            if not single_glob:
                continue
            img = cv2.imread(single_glob[0])
            yaml_path = os.path.join(scen_dir, f"{fid}.yaml")
            params = yaml.safe_load(open(yaml_path))
            if not params.get("vehicles"):
                print(f"{scen_dir} {fid}: negative frame, skipped")
                continue
            K = np.array(params["camera0"]["intrinsic"], dtype=np.float64)
            cam_cords = params["camera0"]["cords"]
            veh = list(params["vehicles"].values())[0]
            corners = box_corners_world(veh["location"], veh["center"], veh["extent"],
                                        veh["angle"][1])
            uv, valid, depth = project(corners, cam_cords, K)
            for i, j in EDGES:
                if not (valid[i] and valid[j]):
                    continue
                p1 = tuple(np.round(uv[i]).astype(int))
                p2 = tuple(np.round(uv[j]).astype(int))
                cv2.line(img, p1, p2, (0, 255, 0), 2)
            dist_m = float(np.linalg.norm(np.mean(corners, axis=0)[:2]))
            label = f"{fid}  dist={dist_m:.2f}m"
            cv2.putText(img, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            scen_label = scen_dir.rstrip('/').split('/')[-2]  # scenario name, not the "1" agent dir
            out_path = os.path.join(args.out, f"{scen_label}_{fid}.png")
            cv2.imwrite(out_path, img)
            print("wrote", out_path)


if __name__ == "__main__":
    main()
