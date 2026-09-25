"""Turn one live (ego, peer) frame pair into HEAL's input, and run the model.

FORMAT: identical to InferenceFront/ (build_inference.py), because the
formatting code is IMPORTED from build_inference.py, not re-derived:
CAR_REGISTRY (agent "1" = qcar-52775, "2" = qcar-52776 -- the ids
patch_real_extrinsic.py keys its calibration on), CAM_PARAMS (real per-car
fisheye K/D + BEV visibility mask) and build_agent_yaml() (x10 world scale,
lidar_pose, camera0 cords/intrinsic, empty vehicles{}). The only change is
undistortion: the same cv2.fisheye model (Knew=K) through a remap table
computed once per car, instead of undistortImage() recomputing it per frame.

HOW HEAL CONSUMES IT: one fixed scenario on tmpfs,

    <live_dir>/heal/qcar_live/1/000000.yaml, 000000_camera0.png, 000000_bev_visibility.png
    <live_dir>/heal/qcar_live/2/...
    <live_dir>/staging/   (files are written here, then os.replace'd into heal/;
                           kept outside heal/ because HEAL treats every
                           folder under its root as a scenario)

HEAL's dataset is built over it ONCE (it only records paths at init) and
re-reads the files on every ds[0], so each new pair is written in place
(atomic os.replace) and ds[0] goes through HEAL's full, unmodified
preprocessing -- plugins (1-camera loader, real extrinsic), LSS geometry,
comm_range, collate. Writing and reading happen in the same thread, so a
read never sees a half-updated pair.
"""
import importlib.util
import json
import os
import shutil
import sys
import time

import cv2
import numpy as np
import torch

SCENARIO = "qcar_live"
FRAME = "000000"


def import_script(path, name):
    """Import a pipeline script by path with a clean argv: those scripts
    parse flags at import time (parse_known_args), and this process's own
    flags must not leak into them."""
    saved = sys.argv
    sys.argv = [path]
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.argv = saved
    return module


class LiveHealInput:
    def __init__(self, build_inference, live_dir):
        self.bi = build_inference
        self.live_dir = live_dir
        self.agent_ids = sorted(self.bi.CAR_REGISTRY)
        self.maps = {}
        if os.path.isdir(live_dir):
            shutil.rmtree(live_dir)
        for aid in self.agent_ids:
            os.makedirs(self._cav_dir(aid))
            vis_dst = os.path.join(self._cav_dir(aid), "%s_bev_visibility.png" % FRAME)
            shutil.copy2(self.bi.CAM_PARAMS[aid]["vis_mask"], vis_dst)
        os.makedirs(os.path.join(live_dir, "staging"))

    @staticmethod
    def heal_root(live_dir):
        return os.path.join(live_dir, "heal")

    def _cav_dir(self, aid):
        return os.path.join(self.heal_root(self.live_dir), SCENARIO, aid)

    def undistort(self, aid, bgr):
        """== build_inference.undistort (cv2.fisheye.undistortImage, Knew=K)."""
        h, w = bgr.shape[:2]
        key = (aid, w, h)
        if key not in self.maps:
            p = self.bi.CAM_PARAMS[aid]
            self.maps[key] = cv2.fisheye.initUndistortRectifyMap(
                p["K"], p["D"], np.eye(3), p["K"], (w, h), cv2.CV_16SC2)
        m1, m2 = self.maps[key]
        return cv2.remap(bgr, m1, m2, interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)

    def write_pair(self, frames):
        """frames: {agent_id: {'front_bgr': ndarray, 'ego_pose': dict}}."""
        staging = os.path.join(self.live_dir, "staging")
        for aid in self.agent_ids:
            f = frames[aid]
            pose = f["ego_pose"]
            me = {"x": pose["x"], "y": pose["y"], "z": pose["z"], "yaw_rad": pose["yaw"]}
            self.bi.build_agent_yaml(me, aid, staging, FRAME)
            os.replace(os.path.join(staging, FRAME + ".yaml"),
                       os.path.join(self._cav_dir(aid), FRAME + ".yaml"))
            tmp_png = os.path.join(staging, FRAME + "_camera0.png")
            # PNG level 1: lossless like the offline pipeline, fast on tmpfs.
            f["front_undist"] = self.undistort(aid, f["front_bgr"])  # what HEAL sees; the viewer reuses it
            if not cv2.imwrite(tmp_png, f["front_undist"], [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                raise IOError("could not write %s" % tmp_png)
            os.replace(tmp_png, os.path.join(self._cav_dir(aid), FRAME + "_camera0.png"))


class LiveModel:
    """Model + dataset built exactly like qcar/eval.py (resolved_hypes.json
    replayed, plugins imported, best checkpoint loaded), pointed at the live
    scenario."""

    def __init__(self, model_dir, live_dir, qcar_conf, opt, amp):
        from opencood.data_utils.datasets import build_dataset
        from opencood.tools import train_utils
        from qcar import config, registry
        import qcar.patches.patch_windows_paths  # noqa: F401 -- no-op off Windows

        self.train_utils = train_utils
        hypes = json.load(open(os.path.join(model_dir, "resolved_hypes.json")))
        config.resolve_modules(opt, hypes, qcar_conf)
        config.apply_overrides(hypes, qcar_conf["overrides"], opt.set)

        # The live scenario is not in the model's assignment json; give it
        # the modality the model was trained with for each agent (taken from
        # the model's own assignment file, so nothing is invented here).
        heter = hypes.get("heter") or {}
        if heter.get("assignment_path"):
            original = json.load(open(heter["assignment_path"]))
            first = original[sorted(original)[0]]
            live_assign = os.path.join(live_dir, "live_modality_assign.json")
            json.dump({SCENARIO: first}, open(live_assign, "w"), indent=2)
            heter["assignment_path"] = live_assign
        hypes["validate_dir"] = LiveHealInput.heal_root(live_dir)
        self.hypes = hypes

        self.ds = build_dataset(hypes, visualize=False, train=False)
        if len(self.ds) != 1:
            raise RuntimeError("live dataset should hold exactly 1 frame, has %d" % len(self.ds))
        self.model = registry.create_model(hypes)
        _, self.model = train_utils.load_saved_model(model_dir, self.model)
        # CUDA only: HEAL's camera encoder (heter_encoders.py) calls .cuda()
        # in its constructor, so a camera model cannot be built on CPU.
        self.device = torch.device("cuda")
        self.model.to(self.device).eval()
        self.amp = bool(amp)

    @torch.inference_mode()
    def infer(self):
        """Read the live pair through HEAL's dataset, run the model, decode.
        Returns (pred_corners [N,8,3] np, scores [N] np, timings dict)."""
        t0 = time.perf_counter()
        batch = self.ds.collate_batch_test([self.ds[0]])
        if batch is None:
            return np.zeros((0, 8, 3)), np.zeros(0), {"preprocess_s": time.perf_counter() - t0}
        batch = self.train_utils.to_device(batch, self.device)
        t1 = time.perf_counter()
        with torch.cuda.amp.autocast(enabled=self.amp):
            out = self.model(batch["ego"])
        out = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
               for k, v in out.items()}
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        boxes, scores, _ = self.ds.post_process(batch, {"ego": out})
        t3 = time.perf_counter()
        boxes = np.zeros((0, 8, 3)) if boxes is None else boxes.cpu().numpy()
        scores = np.zeros(0) if scores is None else scores.cpu().numpy()
        return boxes, scores, {"preprocess_s": t1 - t0, "model_s": t2 - t1,
                               "postprocess_s": t3 - t2,
                               "n_agents_fused": int(batch["ego"]["record_len"].sum())}
