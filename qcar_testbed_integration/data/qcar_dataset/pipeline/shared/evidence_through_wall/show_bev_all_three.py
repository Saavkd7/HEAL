"""BEV (bird's-eye, top-down) plot of frame idx=49: ego, peer, the two
static targets, and the model's ACTUAL predicted box centers -- so all 3
recognized objects are visible at once, independent of what fits inside
any single camera's photo frustum (which is exactly the point: ego's own
front camera literally cannot frame Node 9, but the fused detection still
places a box for it).
"""
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

from build_coop_train_val_dataset import STATIC_TARGETS, SCALE, apply_marker_offset  # noqa: E402

MODEL_DIR = "opencood/logs/HeterBaseline_opv2v_camera_attfuse_qcar_coop_2026_09_14_12_37_23"
OUT_DIR = "qcar_dataset/Inference/evidence_through_wall"
TARGET_IDX = 98

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

        batch_data_dev = train_utils.to_device(batch_data, device)
        output_dict = model(batch_data_dev["ego"])
        pred_box_tensor, pred_score, gt_box_tensor = ds.post_process(batch_data_dev, {"ego": output_dict})
        pred_np = pred_box_tensor.cpu().numpy()
        scores = pred_score.cpu().numpy()
        centers_rel = pred_np.mean(axis=1)  # ego-relative, x10 on xy

        # convert predicted centers (ego-relative x10) back to world (real m) for the plot
        c, s = math.cos(ego["yaw_rad"]), math.sin(ego["yaw_rad"])
        pred_world = []
        for cc in centers_rel:
            rx, ry = cc[0] / SCALE, cc[1] / SCALE
            wx = ego["x"] + c * rx - s * ry
            wy = ego["y"] + s * rx + c * ry
            pred_world.append((wx, wy))

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title("BEV real, frame idx=%d — lo que la fusión reconoce" % idx)

        def draw_car(x, y, yaw, color, label):
            ax.plot(x, y, "o", color=color, markersize=10)
            ax.text(x + 0.05, y + 0.05, label, color=color, fontsize=9, weight="bold")
            ax.arrow(x, y, 0.15 * math.cos(yaw), 0.15 * math.sin(yaw),
                      head_width=0.06, color=color)

        draw_car(ego["x"], ego["y"], ego["yaw_rad"], "blue", "EGO .198")
        draw_car(peer["x"], peer["y"], peer["yaw_rad"], "green", "PEER .158 (real)")
        ax.plot(t3["x"], t3["y"], "s", color="magenta", markersize=10)
        ax.text(t3["x"] + 0.05, t3["y"] + 0.05, "Node9 (real, pared p/.198)", color="magenta", fontsize=9)
        ax.plot(t4["x"], t4["y"], "s", color="orange", markersize=10)
        ax.text(t4["x"] + 0.05, t4["y"] + 0.05, "Node11 (real)", color="orange", fontsize=9)

        for (wx, wy), sc in zip(pred_world, scores):
            ax.plot(wx, wy, "x", color="red", markersize=14, mew=3)
            ax.text(wx + 0.05, wy - 0.10, "pred score=%.2f" % sc, color="red", fontsize=8)

        ax.set_xlabel("x mundo (m)")
        ax.set_ylabel("y mundo (m)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(["(puntos = real, X roja = predicción del modelo)"], loc="upper left", fontsize=7)
        out_path = os.path.join(OUT_DIR, "BEV_ALL_THREE_idx%04d.png" % idx)
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        print("wrote", out_path)
        break
