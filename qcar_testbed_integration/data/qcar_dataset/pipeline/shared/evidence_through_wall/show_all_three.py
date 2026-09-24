"""Show a single frame where all 3 real objects are simultaneously
present as GT for ego (.198): the dynamic peer (.158), the wall-blocked
static (Node 9, detectable ONLY via cooperation), and the FOV-limited
static (Node 11, ego's own camera once in range). Draws the model's REAL
predicted boxes (pred_box_tensor) and labels each by nearest-match to its
known category, so it's clear which box is which -- not asserted, matched.
"""
import json
import math
import os
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/mnt/mainvolume/Backup/Projects/HEAL")
sys.path.insert(0, "/mnt/mainvolume/Backup/Projects/HEAL/qcar_dataset/Inference")
os.chdir("/mnt/mainvolume/Backup/Projects/HEAL")

import qcar.patches.patch_1cam_loader  # noqa: E402,F401
import qcar.patches.patch_real_extrinsic  # noqa: E402,F401
from opencood.data_utils.datasets import build_dataset  # noqa: E402
from opencood.tools import train_utils  # noqa: E402
from qcar.patches.patch_real_extrinsic import CAMERA_PARAMS  # noqa: E402

from build_coop_train_val_dataset import (  # noqa: E402
    STATIC_TARGETS, SCALE, apply_marker_offset,
)

MODEL_DIR = "opencood/logs/HeterBaseline_opv2v_camera_attfuse_qcar_coop_2026_09_14_12_37_23"
OUT_DIR = "qcar_dataset/Inference/evidence_through_wall"
TARGET_IDX = 98
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]
R_CAM, T_CAM = CAMERA_PARAMS["1"]


def project_ego_frame(corners_x10, K):
    pts = corners_x10.copy()
    pts[:, 0] /= SCALE
    pts[:, 1] /= SCALE
    rel = pts - T_CAM
    cam = rel @ R_CAM
    valid = cam[:, 2] > 0.05
    uvw = cam @ K.T
    uv = uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-6, None)
    return uv, valid


hypes = json.load(open(os.path.join(MODEL_DIR, "resolved_hypes.json")))
ds = build_dataset(hypes, visualize=False, train=False)
loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=ds.collate_batch_test)
model = train_utils.create_model(hypes)
_, model = train_utils.load_saved_model(MODEL_DIR, model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

t3 = apply_marker_offset(STATIC_TARGETS["3"])
t4 = apply_marker_offset(STATIC_TARGETS["4"])

with torch.inference_mode():
    for idx, batch_data in enumerate(loader):
        if idx != TARGET_IDX or batch_data is None:
            continue
        base = ds.retrieve_base_data(idx)
        p1 = base["1"]["params"]["lidar_pose"]
        p2 = base["2"]["params"]["lidar_pose"]
        ego = {"x": p1[0] / SCALE, "y": p1[1] / SCALE, "yaw_rad": math.radians(p1[4])}
        peer = {"x": p2[0] / SCALE, "y": p2[1] / SCALE, "yaw_rad": math.radians(p2[4])}

        def rel_of(world):
            c, s = math.cos(ego["yaw_rad"]), math.sin(ego["yaw_rad"])
            dx, dy = world["x"] - ego["x"], world["y"] - ego["y"]
            return (c * dx + s * dy) * SCALE, (-s * dx + c * dy) * SCALE

        expect = {
            "PEER(.158)": rel_of(peer),
            "NODE9_via_wall_coop": rel_of(t3),
            "NODE11_own_fov": rel_of(t4),
        }

        batch_data_dev = train_utils.to_device(batch_data, device)
        output_dict = model(batch_data_dev["ego"])
        pred_box_tensor, pred_score, gt_box_tensor = ds.post_process(batch_data_dev, {"ego": output_dict})
        pred_np = pred_box_tensor.cpu().numpy() if (pred_box_tensor is not None and pred_box_tensor.shape[0] > 0) else np.zeros((0, 8, 3))
        scores_np = pred_score.cpu().numpy() if pred_score is not None else np.zeros((0,))

        img_path = base["1"]["camera_data"][0]
        im = cv2.imread(img_path) if isinstance(img_path, str) else cv2.cvtColor(np.array(img_path), cv2.COLOR_RGB2BGR)
        K = np.array(base["1"]["params"]["camera0"]["intrinsic"], dtype=np.float64)

        print("idx=%d ego_world=(%.3f,%.3f) peer_world=(%.3f,%.3f)" % (idx, ego["x"], ego["y"], peer["x"], peer["y"]))
        print("expected ego-relative centers (x10,y10):")
        for name, (ex, ey) in expect.items():
            print("  %-22s (%.2f, %.2f)" % (name, ex, ey))
        print("\nreal predicted boxes (pred_box_tensor), matched to nearest expected category:")
        centers = pred_np.mean(axis=1) if pred_np.shape[0] > 0 else np.zeros((0, 3))
        labels = []
        for i, c in enumerate(centers):
            best_name, best_d = None, 1e9
            for name, (ex, ey) in expect.items():
                d = ((c[0] - ex) ** 2 + (c[1] - ey) ** 2) ** 0.5
                if d < best_d:
                    best_d, best_name = d, name
            match = best_name if best_d < 3.0 else "UNMATCHED"
            labels.append(match)
            print("  pred#%d center=(%.2f,%.2f) score=%.3f -> nearest=%s (d=%.2f)" %
                  (i, c[0], c[1], scores_np[i], match, best_d))

        colors = {"PEER(.158)": (0, 255, 0), "NODE9_via_wall_coop": (255, 0, 255),
                  "NODE11_own_fov": (0, 200, 255), "UNMATCHED": (128, 128, 128)}
        for corners, label in zip(pred_np, labels):
            center = corners.mean(axis=0)
            if (center[0] ** 2 + center[1] ** 2) ** 0.5 < 1.5:
                continue
            uv, valid = project_ego_frame(corners, K)
            if valid.sum() < 4:
                continue
            color = colors[label]
            for i, j in EDGES:
                if valid[i] and valid[j]:
                    cv2.line(im, tuple(np.round(uv[i]).astype(int)),
                              tuple(np.round(uv[j]).astype(int)), color, 2)
            p0 = tuple(np.round(uv[valid][0]).astype(int)) if valid.any() else (10, 60)
            cv2.putText(im, label, p0, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        cv2.putText(im, "idx=%d  verde=PEER  magenta=NODE9(pared)  cyan=NODE11(directo)" % idx,
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        out_path = os.path.join(OUT_DIR, "ALL_THREE_idx%04d.png" % idx)
        cv2.imwrite(out_path, im)
        print("\nwrote", out_path)
        break
