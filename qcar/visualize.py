"""Draw real model predictions (magenta) vs ground truth (yellow) on the
actual camera image, for a handful of validate frames -- the visual
counterpart to qcar/eval.py's recall/AP numbers.

    python qcar/visualize.py                              # conf.json defaults
    python qcar/visualize.py --model_dir opencood/logs/<run> --split validate --n 8 --out <dir>

Modules are replayed from the run's resolved_hypes.json, like qcar/eval.py.
Box projection always uses qcar.patches.patch_real_extrinsic's CAMERA_PARAMS
(the real QCar camera geometry), whatever plugins the run loaded.
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils

from qcar import config, registry

# 12 edges of a 3D bounding-box cuboid, by corner index (0-3 bottom face,
# 4-7 top face, matching box_corners_world()'s corner ordering).
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),
         (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]


def project_ego_frame(corners_xyz10, K, R_cb, t_cb):
    """corners_xyz10: (8,3) in ego body frame, x/y already x10 model-space,
    z real. Same convention box_corners_world() produces."""
    pts = corners_xyz10.copy()
    pts[:, 0] /= 10.0
    pts[:, 1] /= 10.0
    rel = pts - t_cb
    cam = rel @ R_cb
    valid = cam[:, 2] > 0.05
    uvw = cam @ K.T
    uv = uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-6, None)
    return uv, valid


def draw_boxes(img, corners_batch, color):
    any_drawn = False
    H, W = img["im"].shape[:2]
    for corners in corners_batch:
        # Cooperative-fusion GT/pred can include the EGO'S OWN box (the peer
        # labels ego as its own "vehicle", which lands near the origin +
        # the marker-to-center offset once transformed into ego's own
        # frame -- verified 2026-09-14 by tracing the exact corner values).
        # Real targets never sit this close; skip it, it isn't detectable
        # by a forward camera anyway.
        center = corners.mean(axis=0)
        # self-box magnitude is sqrt(0.0343^2+0.0209^2)=0.040m real (0.40 in
        # x10 space); real close-approach targets go down to ~0.3m real
        # (3.0 in x10 space) per the trajectory data, so 1.5 cleanly
        # separates the two without excluding genuine close targets.
        if (center[0] ** 2 + center[1] ** 2) ** 0.5 < 1.5:
            continue
        R_cb, t_cb = img["R_cb"], img["t_cb"]
        K = img["K"]
        uv, valid = project_ego_frame(corners, K, R_cb, t_cb)
        # Cooperative fusion GT/preds can include targets outside THIS
        # camera's own view (visible to the peer instead). Only draw boxes
        # that genuinely fall inside this frame, not just "in front".
        depth_ok = valid  # already depth > 0.05 from project_ego_frame
        in_frame = (uv[:, 0] > -80) & (uv[:, 0] < W + 80) & \
                   (uv[:, 1] > -80) & (uv[:, 1] < H + 80)
        ok = depth_ok & in_frame
        if ok.sum() < 4:
            continue
        for i, j in EDGES:
            if ok[i] and ok[j]:
                cv2.line(img["im"], tuple(np.round(uv[i]).astype(int)),
                          tuple(np.round(uv[j]).astype(int)), color, 2)
                any_drawn = True
    return any_drawn


def parse_args():
    conf = config.load_conf(config.conf_path_from_argv())
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_dir", default=None,
                    help="run dir (default: conf.json visualize_model_dir)")
    ap.add_argument("--split", choices=["train", "validate", "test"],
                    default=conf["visualize_split"])
    ap.add_argument("--n", type=int, default=conf["visualize_n"],
                    help="number of frames to write")
    ap.add_argument("--out", default=None,
                    help="output dir (default: conf.json visualize_out, "
                         "which may use {model_dir} and {split})")
    config.add_module_args(ap)
    opt = ap.parse_args()
    opt.model_dir = (config.cli_path(opt.model_dir) if opt.model_dir
                     else conf.path_of("visualize_model_dir"))
    opt.out = (config.cli_path(opt.out) if opt.out
               else config.repo_path(conf["visualize_out"].format(
                   model_dir=opt.model_dir, split=opt.split)))
    return opt, conf


def main():
    opt, conf = parse_args()
    os.chdir(config.REPO_ROOT)
    os.makedirs(opt.out, exist_ok=True)

    hypes = json.load(open(os.path.join(opt.model_dir, "resolved_hypes.json")))
    config.resolve_modules(opt, hypes, conf)
    config.apply_overrides(hypes, conf["overrides"], opt.set)
    dir_key = {"train": "root_dir", "validate": "validate_dir", "test": "test_dir"}[opt.split]
    h2 = dict(hypes)
    h2["validate_dir"] = hypes[dir_key]
    ds = build_dataset(h2, visualize=False, train=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=ds.collate_batch_test)

    model = registry.create_model(hypes)
    _, model = train_utils.load_saved_model(opt.model_dir, model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    from qcar.patches.patch_real_extrinsic import CAMERA_PARAMS

    written = 0
    with torch.inference_mode():
        for idx, batch_data in enumerate(loader):
            if written >= opt.n or batch_data is None:
                if written >= opt.n:
                    break
                continue
            batch_data = train_utils.to_device(batch_data, device)
            output_dict = model(batch_data["ego"])
            pred_box_tensor, pred_score, gt_box_tensor = ds.post_process(
                batch_data, {"ego": output_dict})
            if gt_box_tensor is None and pred_box_tensor is None:
                continue

            base = ds.retrieve_base_data(idx)
            ego_cav_id = list(base.keys())[0]
            frame_params = base[ego_cav_id]["params"]
            img_path = None
            for key in base[ego_cav_id]:
                if key == "camera_data":
                    img_path = base[ego_cav_id][key][0]
                    break
            if img_path is None:
                continue
            if isinstance(img_path, str):
                im = cv2.imread(img_path)
            else:
                im = cv2.cvtColor(np.array(img_path), cv2.COLOR_RGB2BGR)  # PIL is RGB, cv2 wants BGR
            if im is None:
                continue
            K = np.array(frame_params["camera0"]["intrinsic"], dtype=np.float64)
            cav_id = str(frame_params.get("qcar_provenance", {}).get("cav_id", ego_cav_id))
            R_cb, t_cb = CAMERA_PARAMS.get(cav_id, CAMERA_PARAMS["1"])

            img_ctx = {"im": im, "K": K, "R_cb": R_cb, "t_cb": t_cb}
            drawn = False
            if gt_box_tensor is not None and gt_box_tensor.shape[0] > 0:
                drawn = draw_boxes(img_ctx, gt_box_tensor.cpu().numpy(), (0, 255, 255)) or drawn  # yellow
            if pred_box_tensor is not None and pred_box_tensor.shape[0] > 0:
                drawn = draw_boxes(img_ctx, pred_box_tensor.cpu().numpy(), (255, 0, 255)) or drawn  # magenta
            if not drawn:
                continue  # target too close to the camera for a stable pinhole projection

            n_gt = gt_box_tensor.shape[0] if gt_box_tensor is not None else 0
            n_pred = pred_box_tensor.shape[0] if pred_box_tensor is not None else 0
            cv2.putText(im, f"idx={idx} gt={n_gt}(amarillo) pred={n_pred}(magenta)",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            out_path = os.path.join(opt.out, f"infer_{idx:04d}.png")
            cv2.imwrite(out_path, im)
            print("wrote", out_path, "gt=", n_gt, "pred=", n_pred)
            written += 1


if __name__ == "__main__":
    main()
