"""Honest collage: sample frames EVENLY across the validate set (not
cherry-picked for looking good), run the real model, draw the REAL
pred_box_tensor (magenta) and gt_box_tensor (yellow) on each frame's real
camera photo, tile into one grid, and print an honest per-frame tally --
including misses and false positives, not just wins.
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
    STATIC_TARGETS, WALL_BLOCKED, SCALE, apply_marker_offset, truly_visible_to_agent,
)

MODEL_DIR = "opencood/logs/HeterBaseline_opv2v_camera_attfuse_qcar_coop_2026_09_14_12_37_23"
OUT_DIR = "qcar_dataset/Inference/evidence_through_wall"
N_SAMPLES = 20  # evenly spaced across the 237 validate frames, NOT hand-picked
DIST_THRESH_X10 = 3.0
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


def draw_all(im, boxes, K, color):
    for corners in boxes:
        center = corners.mean(axis=0)
        if (center[0] ** 2 + center[1] ** 2) ** 0.5 < 1.5:
            continue
        uv, valid = project_ego_frame(corners, K)
        if valid.sum() < 4:
            continue
        for i, j in EDGES:
            if valid[i] and valid[j]:
                cv2.line(im, tuple(np.round(uv[i]).astype(int)),
                          tuple(np.round(uv[j]).astype(int)), color, 2)


hypes = json.load(open(os.path.join(MODEL_DIR, "resolved_hypes.json")))
ds = build_dataset(hypes, visualize=False, train=False)
loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=ds.collate_batch_test)
model = train_utils.create_model(hypes)
_, model = train_utils.load_saved_model(MODEL_DIR, model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

n_total = len(ds)
sample_idxs = sorted(set(int(i * (n_total - 1) / (N_SAMPLES - 1)) for i in range(N_SAMPLES)))
print("sampling %d frames (evenly spaced, 0..%d): %s" % (len(sample_idxs), n_total - 1, sample_idxs))

t3 = apply_marker_offset(STATIC_TARGETS["3"]); t3yaw = math.degrees(t3["yaw_rad"])
t4 = apply_marker_offset(STATIC_TARGETS["4"]); t4yaw = math.degrees(t4["yaw_rad"])

tiles = []
tally = {"peer_hit": 0, "peer_miss": 0, "peer_na": 0,
         "node9_hit": 0, "node9_miss": 0, "node9_na": 0,
         "node11_hit": 0, "node11_miss": 0, "node11_na": 0}

with torch.inference_mode():
    for idx, batch_data in enumerate(loader):
        if idx not in sample_idxs or batch_data is None:
            continue
        base = ds.retrieve_base_data(idx)
        if "1" not in base or "2" not in base:
            continue
        p1 = base["1"]["params"]["lidar_pose"]
        p2 = base["2"]["params"]["lidar_pose"]
        ego = {"x": p1[0] / SCALE, "y": p1[1] / SCALE, "z": 0.0, "yaw_rad": math.radians(p1[4])}
        peer = {"x": p2[0] / SCALE, "y": p2[1] / SCALE, "z": 0.0, "yaw_rad": math.radians(p2[4])}
        peer_yaw_deg = math.degrees(peer["yaw_rad"])

        vis_peer = truly_visible_to_agent(peer["x"], peer["y"], peer["z"], peer_yaw_deg, ego, "1") or \
                   truly_visible_to_agent(peer["x"], peer["y"], peer["z"], peer_yaw_deg, peer, "2")
        vis3 = (("1", "3") not in WALL_BLOCKED and truly_visible_to_agent(t3["x"], t3["y"], t3["z"], t3yaw, ego, "1")) or \
               truly_visible_to_agent(t3["x"], t3["y"], t3["z"], t3yaw, peer, "2")
        vis4 = truly_visible_to_agent(t4["x"], t4["y"], t4["z"], t4yaw, ego, "1") or \
               (("2", "4") not in WALL_BLOCKED and truly_visible_to_agent(t4["x"], t4["y"], t4["z"], t4yaw, peer, "2"))

        def rel_of(w):
            c, s = math.cos(ego["yaw_rad"]), math.sin(ego["yaw_rad"])
            dx, dy = w["x"] - ego["x"], w["y"] - ego["y"]
            return (c * dx + s * dy) * SCALE, (-s * dx + c * dy) * SCALE

        expect = {"PEER": rel_of(peer), "NODE9": rel_of(t3), "NODE11": rel_of(t4)}

        batch_data_dev = train_utils.to_device(batch_data, device)
        output_dict = model(batch_data_dev["ego"])
        pred_box_tensor, pred_score, gt_box_tensor = ds.post_process(batch_data_dev, {"ego": output_dict})
        pred_np = pred_box_tensor.cpu().numpy() if (pred_box_tensor is not None and pred_box_tensor.shape[0] > 0) else np.zeros((0, 8, 3))
        gt_np = gt_box_tensor.cpu().numpy() if (gt_box_tensor is not None and gt_box_tensor.shape[0] > 0) else np.zeros((0, 8, 3))
        centers = pred_np.mean(axis=1) if pred_np.shape[0] > 0 else pred_np[:, 0, :]

        found = set()
        for c in centers:
            if (c[0] ** 2 + c[1] ** 2) ** 0.5 < 1.5:
                continue
            for name, (ex, ey) in expect.items():
                d = ((c[0] - ex) ** 2 + (c[1] - ey) ** 2) ** 0.5
                if d < DIST_THRESH_X10:
                    found.add(name)

        for name, vis, key in (("PEER", vis_peer, "peer"), ("NODE9", vis3, "node9"), ("NODE11", vis4, "node11")):
            if not vis:
                tally[key + "_na"] += 1
            elif name in found:
                tally[key + "_hit"] += 1
            else:
                tally[key + "_miss"] += 1

        img_path = base["1"]["camera_data"][0]
        im = cv2.imread(img_path) if isinstance(img_path, str) else cv2.cvtColor(np.array(img_path), cv2.COLOR_RGB2BGR)
        K = np.array(base["1"]["params"]["camera0"]["intrinsic"], dtype=np.float64)
        draw_all(im, gt_np, K, (0, 255, 255))
        draw_all(im, pred_np, K, (255, 0, 255))
        cv2.putText(im, "idx=%d" % idx, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        tiles.append((idx, cv2.resize(im, (320, 240))))
        print("idx=%d  vis(peer=%s,node9=%s,node11=%s)  found=%s  n_pred=%d n_gt=%d" %
              (idx, vis_peer, vis3, vis4, sorted(found), pred_np.shape[0], gt_np.shape[0]))

# tile into a grid, 5 columns
cols = 5
rows = (len(tiles) + cols - 1) // cols
grid = np.full((rows * 240, cols * 320, 3), 30, dtype=np.uint8)
for i, (idx, im) in enumerate(tiles):
    r, c = divmod(i, cols)
    grid[r * 240:(r + 1) * 240, c * 320:(c + 1) * 320] = im
out_path = os.path.join(OUT_DIR, "COLLAGE_HONEST_%dframes.png" % len(tiles))
cv2.imwrite(out_path, grid)
print("\nwrote", out_path)
print("\nTALLY (solo frames donde el target realmente era visible a alguien -- 'na' = nadie lo veia, no cuenta):")
for k in ("peer", "node9", "node11"):
    hit, miss, na = tally[k + "_hit"], tally[k + "_miss"], tally[k + "_na"]
    tot = hit + miss
    rate = "%.0f%%" % (100 * hit / tot) if tot else "n/a"
    print("  %-8s hit=%d miss=%d (na=%d)  recall=%s" % (k, hit, miss, na, rate))
