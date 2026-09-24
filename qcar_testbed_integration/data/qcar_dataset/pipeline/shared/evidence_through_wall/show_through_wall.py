"""Visual proof (or disproof) of "sees through the wall" -- CORRECTED
version. Draws the model's ACTUAL predicted boxes (straight from
pred_box_tensor, at their own predicted position/size/orientation -- NOT
snapped to the known Vicon location) in magenta, and the actual GT boxes
in yellow, for EVERY object in the frame (not just target 3) -- so what's
on screen is honestly what the network output, full context included.
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
    STATIC_TARGETS, SCALE, apply_marker_offset, truly_visible_to_agent,
)

MODEL_DIR = "opencood/logs/HeterBaseline_opv2v_camera_attfuse_qcar_coop_2026_09_14_12_37_23"
OUT = "/tmp/claude-1000/-mnt-mainvolume-Backup-Projects-Concordia/4ce9f3ce-b9d2-4763-81c4-54ec1bc00e12/scratchpad/through_wall_v2"
os.makedirs(OUT, exist_ok=True)

DIST_THRESH_X10 = 3.0
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]

hypes = json.load(open(os.path.join(MODEL_DIR, "resolved_hypes.json")))
ds = build_dataset(hypes, visualize=False, train=False)
loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=ds.collate_batch_test)
model = train_utils.create_model(hypes)
_, model = train_utils.load_saved_model(MODEL_DIR, model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

target3_world = apply_marker_offset(STATIC_TARGETS["3"])
tgt_yaw_deg = math.degrees(target3_world["yaw_rad"])

R_CAM, T_CAM = CAMERA_PARAMS["1"]


def project_ego_frame(corners_x10, K):
    """corners_x10: (8,3), x/y already x10 ego-relative (pred/gt tensor's
    own convention), z real -- SAME box the network/eval code actually
    produced, only reprojected into pixels for display."""
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
            continue  # self-box artifact, not a real target
        uv, valid = project_ego_frame(corners, K)
        if valid.sum() < 4:
            continue
        for i, j in EDGES:
            if valid[i] and valid[j]:
                cv2.line(im, tuple(np.round(uv[i]).astype(int)),
                          tuple(np.round(uv[j]).astype(int)), color, 2)


results = []
with torch.inference_mode():
    for idx, batch_data in enumerate(loader):
        if batch_data is None:
            continue
        base = ds.retrieve_base_data(idx)
        if "1" not in base or "2" not in base:
            continue
        p1 = base["1"]["params"]["lidar_pose"]
        p2 = base["2"]["params"]["lidar_pose"]
        ego = {"x": p1[0] / SCALE, "y": p1[1] / SCALE, "z": 0.0, "yaw_rad": math.radians(p1[4])}
        peer = {"x": p2[0] / SCALE, "y": p2[1] / SCALE, "z": 0.0, "yaw_rad": math.radians(p2[4])}
        peer_sees_3 = truly_visible_to_agent(
            target3_world["x"], target3_world["y"], target3_world["z"], tgt_yaw_deg, peer, "2")

        batch_data_dev = train_utils.to_device(batch_data, device)
        output_dict = model(batch_data_dev["ego"])
        pred_box_tensor, pred_score, gt_box_tensor = ds.post_process(batch_data_dev, {"ego": output_dict})
        pred_np = pred_box_tensor.cpu().numpy() if (pred_box_tensor is not None and pred_box_tensor.shape[0] > 0) else np.zeros((0, 8, 3))
        gt_np = gt_box_tensor.cpu().numpy() if (gt_box_tensor is not None and gt_box_tensor.shape[0] > 0) else np.zeros((0, 8, 3))

        ex, ey, eyaw = ego["x"], ego["y"], ego["yaw_rad"]
        c, s = math.cos(eyaw), math.sin(eyaw)
        dx, dy = target3_world["x"] - ex, target3_world["y"] - ey
        rel_x = (c * dx + s * dy) * SCALE
        rel_y = (-s * dx + c * dy) * SCALE
        hit3 = False
        if pred_np.shape[0] > 0:
            centers = pred_np.mean(axis=1)
            d = np.linalg.norm(centers[:, :2] - np.array([rel_x, rel_y]), axis=1)
            hit3 = bool(d.min() < DIST_THRESH_X10)

        results.append({"idx": idx, "peer_sees_3": peer_sees_3, "hit3": hit3,
                        "n_pred": pred_np.shape[0], "n_gt": gt_np.shape[0],
                        "pred_np": pred_np, "gt_np": gt_np})

coop_examples = [r for r in results if r["peer_sees_3"] and r["hit3"]]
blind_examples = [r for r in results if not r["peer_sees_3"] and not r["hit3"]]
print("cooperation-working examples:", len(coop_examples))
print("correctly-empty (peer blind) examples:", len(blind_examples))

written = []
for tag, subset in (("COOP_WORKING", coop_examples[:4]), ("PEER_BLIND_CORRECT", blind_examples[:3])):
    for r in subset:
        idx = r["idx"]
        base = ds.retrieve_base_data(idx)
        img_path = base["1"]["camera_data"][0]
        im = cv2.imread(img_path) if isinstance(img_path, str) else cv2.cvtColor(np.array(img_path), cv2.COLOR_RGB2BGR)
        K = np.array(base["1"]["params"]["camera0"]["intrinsic"], dtype=np.float64)
        draw_all(im, r["gt_np"], K, (0, 255, 255))   # yellow = real GT (all objects)
        draw_all(im, r["pred_np"], K, (255, 0, 255))  # magenta = model's ACTUAL predicted boxes
        cv2.putText(im, "%s idx=%d n_pred=%d n_gt=%d peer_sees_node9=%s" %
                    (tag, idx, r["n_pred"], r["n_gt"], r["peer_sees_3"]),
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        out_path = os.path.join(OUT, "%s_idx%04d.png" % (tag, idx))
        cv2.imwrite(out_path, im)
        written.append(out_path)

print("\nwrote:")
for w in written:
    print(" ", w)
