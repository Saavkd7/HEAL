"""Round-by-round cooperative inference on the QCar dataset -- "real time" on a PC.

Each round is one frame of the dataset, in temporal order. The EGO runs live
(camera encoder -> BEV backbone). Each PEER does not: its transmitted feature
-- the per-agent BEV feature right before fusion, which is what HEAL sends
between agents -- is precomputed once per method and loaded from disk as if it
had just arrived over the network. The ego warps it into its own frame (the
fusion module does this with the pairwise pose), fuses, runs the detection
head, and the boxes are scored against GT with running metrics. The same round
is also run with the ego's feature alone, to show what cooperation adds.

A model is split into four stages, matching HEAL's own forward():

    encoder       encoder_m*                          image -> BEV (LSS)
    bev_backbone  backbone_m*, shrinker_m*/aligner_m*, compressor
                                                      BEV -> transmitted feature
    fusion        fusion_net / pyramid_backbone / backbone (multiscale)
    head          shrink_conv, cls/reg/dir heads      fused BEV -> boxes

The architecture always comes from the FUSION method's zoo run. By default every
stage's weights come from that same checkpoint -- each zoo model was trained end
to end, so its encoder only speaks its own fusion's "language". conf.json
rounds_stage_sources can load a stage from another zoo run instead; that only
loads when the tensors match exactly, and warns, because stages trained apart
need alignment first (HEAL's stage 2).

Agents are encoded one at a time, as each car would on its own hardware. That
differs from HEAL's batched forward() by ~5e-5 relative at the encoder output:
LSS's cumsum-trick pooling (QuickCumsum) rounds differently when other agents'
points share the batch (an exact segment sum brings it to 1.9e-7), plus TF32
convolutions. AP on the full validation split is identical to compare.py.

Everything is set in conf.json (rounds_* keys, plus compare_zoo_dir); the
notebook qcar_rounds.ipynb at the repo root drives this module. Only camera
agents are supported (the zoo is camera-only today).
"""
import argparse
import glob
import json
import os
import shutil
import time
import warnings
from collections import Counter

import numpy as np
import torch
import torchvision
from torch import nn

from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
from opencood.tools import train_utils
from opencood.utils import eval_utils
from opencood.utils.transformation_utils import normalize_pairwise_tfm

from qcar import compare, config, registry

STAGES = ("encoder", "bev_backbone", "fusion", "head")

# Every checkpoint key belongs to exactly one stage, by prefix. Order matters:
# "backbone_m2." (per-modality BEV backbone) before "backbone." (the multiscale
# fusion backbone of heter_model_baseline_ms).
STAGE_PREFIXES = (
    ("encoder", ("encoder_",)),
    ("bev_backbone", ("backbone_m", "shrinker_", "aligner_", "compressor.")),
    ("fusion", ("fusion_net.", "pyramid_backbone.", "backbone.")),
    ("head", ("shrink_conv.", "cls_head", "reg_head", "dir_head")),
)

NO_CLI = argparse.Namespace(plugins=None, model_packages=None, loss_packages=None)


def stage_of(key):
    for stage, prefixes in STAGE_PREFIXES:
        if key.startswith(prefixes):
            return stage
    raise KeyError("checkpoint key %r belongs to no stage -- extend STAGE_PREFIXES" % key)


def best_checkpoint(method_dir):
    found = glob.glob(os.path.join(method_dir, "net_epoch_bestval_at*.pth"))
    if len(found) != 1:
        raise SystemExit("expected one net_epoch_bestval_at*.pth in %s, found %d"
                         % (method_dir, len(found)))
    return found[0]


def zoo_methods(conf):
    zoo = conf.path_of("compare_zoo_dir")
    return sorted(d for d in os.listdir(zoo)
                  if os.path.isfile(os.path.join(zoo, d, "resolved_hypes.json")))


def stage_sources(method, overrides):
    """{stage: zoo method whose weights fill it}. fusion is always `method`."""
    unknown = set(overrides) - {"encoder", "bev_backbone", "head"}
    if unknown:
        raise SystemExit("rounds_stage_sources: unknown or fixed stages %s "
                         "(fusion is always rounds_method)" % sorted(unknown))
    return {s: overrides.get(s) or method for s in STAGES}


def build_model(conf, method, sources, device):
    """The fusion method's architecture, each stage loaded from its source run."""
    zoo = conf.path_of("compare_zoo_dir")
    hypes = json.load(open(os.path.join(zoo, method, "resolved_hypes.json")))
    config.resolve_modules(NO_CLI, hypes, conf)
    model = registry.create_model(hypes)

    ckpts = {src: best_checkpoint(os.path.join(zoo, src)) for src in set(sources.values())}
    states = {src: torch.load(path, map_location="cpu") for src, path in ckpts.items()}
    merged, bad = {}, []
    for key, target in model.state_dict().items():
        src = sources[stage_of(key)]
        tensor = states[src].get(key)
        if tensor is None or tensor.shape != target.shape:
            bad.append("%s from %s: %s" % (key, src, "missing" if tensor is None
                                           else "shape %s != %s" % (tuple(tensor.shape),
                                                                    tuple(target.shape))))
        else:
            merged[key] = tensor
    if bad:
        raise SystemExit("stage weights do not fit the %s architecture (%d tensors), e.g.\n  %s"
                         % (method, len(bad), "\n  ".join(bad[:5])))
    model.load_state_dict(merged, strict=True)
    for stage, src in sources.items():
        if src != method:
            warnings.warn("stage %s loaded from %s but fusion is %s: they were trained "
                          "apart, so features may not align (HEAL stage-2 problem)"
                          % (stage, src, method))
    return hypes, model.to(device).eval(), ckpts


def agent_inputs(data_dict, modality, index):
    """data_dict narrowed to the index-th agent of `modality`, as its encoder reads it."""
    inputs = data_dict["inputs_" + modality]
    if "imgs" not in inputs:
        raise NotImplementedError("per-agent slicing is only implemented for camera "
                                  "inputs; lidar voxels need batch-index remapping")
    return {"inputs_" + modality: {k: v[index:index + 1] for k, v in inputs.items()}}


def encode(model, data_dict, agents):
    """Run encoder + bev_backbone for the given agent indices, one agent at a time
    (as each car would on its own hardware).

    Returns (raw, feats): raw[i] = agent's encoder BEV (1,C,H,W), feats = the
    stacked transmitted features (len(agents),C,H,W) -- identical to the
    heter_feature_2d HEAL builds inside forward() before fusing.
    """
    modality_list = data_dict["agent_modality_list"]
    seen, within = Counter(), []
    for m in modality_list:
        within.append(seen[m])
        seen[m] += 1

    raw, feats = [], []
    for a in agents:
        m = modality_list[a]
        x = getattr(model, "encoder_" + m)(agent_inputs(data_dict, m, within[a]), m)
        raw.append(x)
        x = getattr(model, "backbone_" + m)({"spatial_features": x})["spatial_features_2d"]
        adapter = "shrinker_" + m if hasattr(model, "shrinker_" + m) else "aligner_" + m
        x = getattr(model, adapter)(x)
        if model.sensor_type_dict[m] == "camera":
            _, _, H, W = x.shape
            x = torchvision.transforms.CenterCrop(
                (int(H * getattr(model, "crop_ratio_H_" + m)),
                 int(W * getattr(model, "crop_ratio_W_" + m))))(x)
        feats.append(x)
    feats = torch.cat(feats)
    if getattr(model, "compress", False):
        feats = model.compressor(feats)
    return raw, feats


def fuse(model, feats, affine, modality_list):
    """The fusion stage of each of the three HEAL model families."""
    record_len = torch.tensor([len(feats)], device=feats.device)
    if hasattr(model, "pyramid_backbone"):  # heter_pyramid_collab
        fused, _ = model.pyramid_backbone.forward_collab(
            feats, record_len, affine, modality_list, model.cam_crop_info)
        return fused
    if isinstance(model.fusion_net, nn.ModuleList):  # heter_model_baseline_ms
        levels = [feats]
        for i in range(1, len(model.fusion_net)):
            levels.append(model.backbone.get_layer_i_feature(levels[-1], layer_i=i))
        return model.backbone.decode_multiscale_feature(
            [f(x, record_len, affine) for f, x in zip(model.fusion_net, levels)])
    return model.fusion_net(feats, record_len, affine)  # heter_model_baseline


def head(model, fused):
    if model.shrink_flag:
        fused = model.shrink_conv(fused)
    return {"cls_preds": model.cls_head(fused), "reg_preds": model.reg_head(fused),
            "dir_preds": model.dir_head(fused)}


def affine_for(model, data_dict):
    """Normalized pairwise transforms, padded to max_cav exactly as forward()
    passes them: fusions take [:N, :N] themselves (ego first), and some
    (CoBEVT) regroup to the padded size, so never slice this down."""
    return normalize_pairwise_tfm(data_dict["pairwise_t_matrix"], model.H, model.W,
                                  model.fake_voxel_size)


def peer_cache_dir(conf, method, sources, split):
    return os.path.join(conf.path_of("rounds_cache_dir"), split,
                        "%s__enc-%s__bev-%s" % (method, sources["encoder"],
                                                 sources["bev_backbone"]))


def build_peer_cache(model, ds, frames, out_dir, manifest, dtype):
    """Encode every peer of every frame once, and store what each would transmit."""
    os.makedirs(out_dir, exist_ok=True)
    with torch.inference_mode():
        for n, idx in enumerate(frames):
            batch = train_utils.to_device(ds.collate_batch_test([ds[idx]]), model_device(model))
            ego = batch["ego"]
            peers = list(range(1, len(ego["agent_modality_list"])))
            feats = encode(model, ego, peers)[1] if peers else None
            torch.save({"cav_ids": ego["cav_id_list"][1:],
                        "features": None if feats is None else feats.to(dtype).cpu()},
                       os.path.join(out_dir, "frame_%05d.pt" % idx))
            if n % 25 == 0:
                print("peer cache %d/%d" % (n, len(frames)), flush=True)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def model_device(model):
    return next(model.parameters()).device


def new_stat(ious):
    return {iou: {"tp": [], "fp": [], "score": [], "gt": 0} for iou in ious}


def merge_stats(parts, ious):
    total = new_stat(ious)
    for part in parts:
        for iou in ious:
            for k in ("tp", "fp", "score"):
                total[iou][k] += part[iou][k]
            total[iou]["gt"] += part[iou]["gt"]
    return total


def summarize(stat, ious):
    out = {}
    for iou in ious:
        s = stat[iou]
        tp = int(np.sum(s["tp"])) if s["tp"] else 0
        fp = int(np.sum(s["fp"])) if s["fp"] else 0
        ap = float(eval_utils.calculate_ap(stat, iou)[0]) if s["gt"] and s["tp"] else 0.0
        out[iou] = {"tp": tp, "fp": fp, "gt": s["gt"], "ap": ap,
                    "recall": tp / s["gt"] if s["gt"] else float("nan")}
    return out


class Rounds:
    """One session: a method (+ optional stage sources) over a sequence of frames.

    rounds[r] runs round r and returns a dict with images, BEV maps, boxes and
    metrics; the running metrics cover every round run so far (re-running a
    round does not double count it).
    """

    def __init__(self, conf, method=None):
        self.conf = conf
        self.method = method or conf["rounds_method"]
        self.split = conf["rounds_split"]
        self.ious = [float(i) for i in conf["rounds_iou"]]
        self.sources = stage_sources(self.method, conf["rounds_stage_sources"])
        self.device = torch.device(conf["rounds_device"])  # HEAL's LSS encoder needs CUDA
        os.chdir(config.REPO_ROOT)  # dataset paths in the hypes are repo-relative
        self.hypes, self.model, ckpts = build_model(conf, self.method, self.sources,
                                                    self.device)
        self.ds = compare.build_split(self.hypes, self.split)
        self.lidar_range = self.hypes["preprocess"]["cav_lidar_range"]

        traj, names = compare.frame_trajectory(self.ds, len(self.ds))
        wanted = conf["rounds_scenarios"] or names
        missing = set(wanted) - set(names)
        if missing:
            raise SystemExit("rounds_scenarios %s not in split %s (has %s)"
                             % (sorted(missing), self.split, names))
        self.frames = [i for i in range(len(self.ds)) if names[traj[i]] in wanted]
        self.scenario_of = {i: names[traj[i]] for i in self.frames}

        self.cache_dtype = getattr(torch, conf["rounds_cache_dtype"])
        self.cache_dir = peer_cache_dir(conf, self.method, self.sources, self.split)
        manifest = {"method": self.method, "split": self.split, "frames": self.frames,
                    "dtype": conf["rounds_cache_dtype"],
                    "checkpoints": {s: os.path.relpath(ckpts[self.sources[s]], config.REPO_ROOT)
                                    for s in ("encoder", "bev_backbone")}}
        manifest_path = os.path.join(self.cache_dir, "manifest.json")
        cached = json.load(open(manifest_path)) if os.path.isfile(manifest_path) else None
        if cached is None or {k: cached.get(k) for k in manifest} != manifest:
            if not conf["rounds_cache_keep_other_methods"]:
                split_dir = os.path.dirname(self.cache_dir)
                for other in glob.glob(os.path.join(split_dir, "*__enc-*__bev-*")):
                    if other != self.cache_dir:
                        print("removing other method's peer cache %s" % other, flush=True)
                        shutil.rmtree(other)
            print("building peer feature cache -> %s" % self.cache_dir, flush=True)
            build_peer_cache(self.model, self.ds, self.frames, self.cache_dir, manifest,
                             self.cache_dtype)
        self.done = {}  # round -> {"coop": stat, "ego": stat}

    def __len__(self):
        return len(self.frames)

    def load_peer(self, idx, cav_ids):
        blob = torch.load(os.path.join(self.cache_dir, "frame_%05d.pt" % idx))
        if blob["cav_ids"] != cav_ids:
            raise RuntimeError("peer cache for frame %d has cavs %s, dataset has %s -- "
                               "delete %s to rebuild" % (idx, blob["cav_ids"], cav_ids,
                                                         self.cache_dir))
        return blob["features"]

    def __getitem__(self, r):
        idx = self.frames[r]
        amp = self.conf["rounds_amp"] and self.device.type == "cuda"
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=amp):
            batch = train_utils.to_device(self.ds.collate_batch_test([self.ds[idx]]),
                                          self.device)
            ego = batch["ego"]
            modalities = ego["agent_modality_list"]

            t0 = time.perf_counter()
            _, feat_ego = encode(self.model, ego, [0])
            self._sync()
            t_encode = time.perf_counter() - t0

            peer = self.load_peer(idx, ego["cav_id_list"][1:])
            peer = None if peer is None else peer.to(self.device, feat_ego.dtype)
            feats = feat_ego if peer is None else torch.cat([feat_ego, peer])

            t0 = time.perf_counter()
            affine = affine_for(self.model, ego)
            grab = {}  # the fused BEV as the head sees it (after shrink_conv)
            hook = self.model.cls_head.register_forward_hook(
                lambda m, i, o: grab.update(x=i[0]))
            try:
                coop_out = head(self.model, fuse(self.model, feats, affine, modalities))
            finally:
                hook.remove()
            self._sync()
            t_fuse = time.perf_counter() - t0
            ego_out = head(self.model, fuse(self.model, feat_ego, affine, modalities[:1]))

            warped = warp_affine_simple(feats.float(), affine[0, 0, :len(feats)],
                                        feats.shape[-2:])
            coop = self._boxes(batch, coop_out)
            solo = self._boxes(batch, ego_out)

        gt = coop["gt"]
        self.done[r] = {"coop": self._stat(coop), "ego": self._stat(solo)}
        seen = list(self.done.values())
        return {
            "round": r, "n_rounds": len(self), "frame": idx,
            "scenario": self.scenario_of[idx], "cav_ids": ego["cav_id_list"],
            "images": self._images(idx),
            "ego_to_agent": ego["pairwise_t_matrix"][0, 0].cpu().numpy(),
            "bev": {"ego": self._energy(warped[0]),
                    "peers": [self._energy(w) for w in warped[1:]],
                    "fused": self._energy(grab["x"][0]),
                    "detection": torch.sigmoid(coop_out["cls_preds"][0].float())
                    .max(0)[0].cpu().numpy()},
            "boxes": {"gt": gt, "coop": coop["pred"], "coop_score": coop["score"],
                      "ego": solo["pred"], "ego_score": solo["score"]},
            "round_metrics": {m: summarize(s, self.ious) for m, s in self.done[r].items()},
            "running_metrics": {m: summarize(merge_stats([d[m] for d in seen], self.ious),
                                             self.ious) for m in ("coop", "ego")},
            "rounds_seen": len(seen),
            "timing_ms": {"ego_encode": 1e3 * t_encode, "fuse_head": 1e3 * t_fuse},
            "transmitted_kb": 0 if peer is None else
            peer[0].numel() * torch.finfo(self.cache_dtype).bits / 8 / 1024,
            "lidar_range": self.lidar_range,
        }

    def reset_metrics(self):
        self.done = {}

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    @staticmethod
    def _energy(feature):
        """(C,H,W) feature -> (H,W) L2 norm over channels, for display."""
        return feature.float().norm(dim=0).cpu().numpy()

    def _boxes(self, batch, out):
        pred, score, gt = self.ds.post_process(batch, {"ego": out})
        as_np = lambda t: np.zeros((0, 8, 3)) if t is None else t.float().cpu().numpy()
        return {"pred": as_np(pred), "gt": as_np(gt), "pred_t": pred, "score_t": score,
                "gt_t": gt, "score": np.zeros(0) if score is None
                else score.float().cpu().numpy()}

    def _stat(self, boxes):
        stat = new_stat(self.ious)
        if boxes["gt_t"] is None:
            return stat
        for iou in self.ious:
            eval_utils.caluclate_tp_fp(boxes["pred_t"], boxes["score_t"], boxes["gt_t"],
                                       stat, iou)
        return stat

    def _images(self, idx):
        from qcar.patches.patch_real_extrinsic import CAMERA_PARAMS
        out = []
        for cav, data in self.ds.retrieve_base_data(idx).items():
            params = data["params"]
            img = data["camera_data"][0]
            img = np.array(img.convert("RGB")) if hasattr(img, "convert") else img
            qcar_id = str(params.get("qcar_provenance", {}).get("cav_id", cav))
            R_cb, t_cb = CAMERA_PARAMS[qcar_id]
            out.append({"cav": cav, "qcar_id": qcar_id, "im": img,
                        "K": np.array(params["camera0"]["intrinsic"], dtype=np.float64),
                        "R_cb": R_cb, "t_cb": t_cb})
        return out


# ---------------------------------------------------------------- drawing --

M_PER_UNIT = 0.1  # model space is x10 the real world in x/y
COLORS = {"gt": (255, 215, 0), "coop": (255, 0, 255), "ego": (0, 200, 255)}  # RGB


def to_agent_frame(corners, ego_to_agent):
    """(N,8,3) ego-frame corners -> the agent's frame (both in x10 model space)."""
    if len(corners) == 0:
        return corners
    homo = np.concatenate([corners, np.ones(corners.shape[:2] + (1,))], axis=2)
    return (homo @ ego_to_agent.T)[..., :3]


def camera_view(image, boxes_by_color):
    from qcar.visualize import draw_boxes
    view = dict(image, im=image["im"].copy())
    for color, corners in boxes_by_color:
        draw_boxes(view, corners, color)
    return view["im"]


def draw(result, fig):
    """Render one round onto a matplotlib figure (cleared first)."""
    fig.clf()
    grid = fig.add_gridspec(2, 4, height_ratios=[1.15, 1], hspace=0.28, wspace=0.12)
    boxes = result["boxes"]

    for i, image in enumerate(result["images"][:2]):
        T = result["ego_to_agent"][i]
        ax = fig.add_subplot(grid[0, 2 * i:2 * i + 2])
        ax.imshow(camera_view(image, [(COLORS["gt"], to_agent_frame(boxes["gt"], T)),
                                      (COLORS["coop"], to_agent_frame(boxes["coop"], T))]))
        ax.set_title("%s camera -- cav %s (QCar %s)" % ("EGO" if i == 0 else "PEER",
                                                        image["cav"], image["qcar_id"]))
        ax.axis("off")

    x0, y0, _, x1, y1, _ = [v * M_PER_UNIT for v in result["lidar_range"]]
    agents = [np.linalg.inv(T)[:2, 3] * M_PER_UNIT for T in result["ego_to_agent"]]
    bev = result["bev"]
    panels = [("ego feature (transmitted)", bev["ego"], [("gt", boxes["gt"]), ("ego", boxes["ego"])]),
              ("peer feature, warped to ego", bev["peers"][0] if bev["peers"] else None,
               [("gt", boxes["gt"])]),
              ("fused feature", bev["fused"], [("gt", boxes["gt"]), ("coop", boxes["coop"])]),
              ("detection score (coop)", bev["detection"], [("gt", boxes["gt"]), ("coop", boxes["coop"])])]
    for k, (title, grid_map, overlays) in enumerate(panels):
        ax = fig.add_subplot(grid[1, k])
        if grid_map is None:
            ax.text(0.5, 0.5, "no peer in range", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.imshow(grid_map, origin="lower", extent=[x0, x1, y0, y1], cmap="magma",
                      vmin=0, vmax=1 if k == 3 else None)
        for name, corners in overlays:
            for c in corners:
                ring = np.vstack([c[:4, :2], c[:1, :2]]) * M_PER_UNIT
                ax.plot(ring[:, 0], ring[:, 1], color=np.array(COLORS[name]) / 255, lw=1.4)
        for j, (ax_, ay_) in enumerate(agents):
            ax.plot(ax_, ay_, marker="^" if j == 0 else "s", color="lime" if j == 0 else "white",
                    ms=7, mec="black")
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x (m)", fontsize=8)
        ax.tick_params(labelsize=7)

    rm, run = result["round_metrics"], result["running_metrics"]
    ious = sorted(run["coop"])
    this = "   ".join("%s TP %d FP %d / GT %d" % (label, rm[m][ious[0]]["tp"], rm[m][ious[0]]["fp"],
                                                    rm[m][ious[0]]["gt"])
                      for m, label in (("coop", "coop"), ("ego", "ego-only")))
    running = "   ".join("AP@%.1f coop %.3f vs ego-only %.3f (recall %.2f vs %.2f)"
                         % (iou, run["coop"][iou]["ap"], run["ego"][iou]["ap"],
                            run["coop"][iou]["recall"], run["ego"][iou]["recall"])
                         for iou in ious)
    fig.suptitle(
        "round %d/%d   %s  frame %d   |   this round @IoU %.1f: %s\n"
        "running over %d rounds: %s\n"
        "ego encode %.0f ms, fuse+head %.0f ms, peer sends %.0f KB   |   "
        "yellow = GT, magenta = coop prediction, cyan = ego-only prediction, "
        "green triangle = ego, white square = peer"
        % (result["round"] + 1, result["n_rounds"], result["scenario"], result["frame"],
           ious[0], this, result["rounds_seen"], running, result["timing_ms"]["ego_encode"],
           result["timing_ms"]["fuse_head"], result["transmitted_kb"]),
        fontsize=9.5, y=0.995)
