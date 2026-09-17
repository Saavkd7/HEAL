"""Training script for the QCar-real, camera-only, front-only fine-tuning run.

Built on HEAL's own opencood/tools/train.py machinery (create_model,
create_loss, setup_optimizer, setup_lr_schedular, setup_train, to_device) --
NOT a reimplementation. The differences from stock train.py, and why each
one exists, are documented inline at the point they matter. Nothing in
opencood/ itself is edited; the 1-camera loader patch is applied by importing
opencood.qcar_patches.patch_1cam_loader, matching the "port, don't modify"
discipline used for every other HEAL/QuantV2X fork in this project.

WHY A SEPARATE SCRIPT INSTEAD OF STOCK train.py's CLI
--------------------------------------------------------
1. Stock train.py unconditionally shells out to inference.py at the end,
   which imports open3d (LiDAR visualisation) -- an unnecessary, possibly
   unavailable dependency for a camera-only run. This script calls the
   lean, open3d-free eval_qcar.py instead (see that file).
2. Stock train.py skips any batch without a positive object.  That wastes the
   QCar negative frames even though PointPillarDepthLoss clamps the positive
   normalizer and computes a valid classification loss for negative-only
   batches.  This script retains those batches so false-positive suppression
   can learn from them; only a genuinely missing batch is skipped.
3. A fixed seed is set once, at the very top, covering torch/cuda/numpy/
   python's random -- the project's own INV-14 discipline (content-
   invariants.md), applied here even though this is not an R/Julia script.
4. The exact resolved hypes (every override applied) are dumped next to the
   checkpoints on every run, so a reviewer can reconstruct precisely what
   config produced a given result without trusting a verbal description.

WHAT IS DELIBERATELY NOT DONE HERE
-------------------------------------
No training run is executed by writing this file. The current development
split can select checkpoints and validate the full pipeline, but it is not an
independent final test. Do not report its numbers as a final QCar detection
result without first freezing the protocol and collecting another trajectory.
"""
from __future__ import print_function

import argparse
import glob
import json
import os
import random
import statistics
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import opencood.qcar_patches.patch_1cam_loader  # noqa: F401  (activates 1-cam file discovery)
import opencood.qcar_patches.patch_real_extrinsic  # noqa: F401  (real camera rotation, not CARLA's mirror)
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils

SEED = 42


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hypes_yaml", "-y", required=True,
                    help="e.g. opencood/hypes_yaml/opv2v/CameraOnly/qcar_real/"
                         "camera_attfuse_onlyfront.yaml")
    ap.add_argument("--model_dir", default="",
                    help="resume from this checkpoint dir instead of training from scratch")
    ap.add_argument("--batch_size", type=int, default=1,
                    help="verified safe micro-batch for the 3.68 GiB QCar GPU")
    ap.add_argument("--accumulation_steps", type=int, default=2,
                    help="gradient accumulation; default gives effective batch size 2")
    ap.add_argument("--gradient_clip", type=float, default=10.0)
    ap.add_argument("--no_amp", action="store_true",
                    help="disable CUDA mixed precision (enabled by default on CUDA)")
    ap.add_argument("--epoches", type=int, default=None,
                    help="override train_params.epoches from the yaml")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--preflight_only", action="store_true",
                    help="validate dataset/splits and exit before model or GPU setup")
    return ap.parse_args()


def main():
    opt = parse_args()
    set_all_seeds(opt.seed)

    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    if opt.epoches:
        hypes["train_params"]["epoches"] = opt.epoches
    hypes["train_params"]["batch_size"] = opt.batch_size
    hypes["_qcar_seed"] = opt.seed
    hypes["_qcar_train_script"] = os.path.abspath(__file__)
    hypes["_qcar_gradient_accumulation_steps"] = opt.accumulation_steps
    hypes["_qcar_gradient_clip"] = opt.gradient_clip
    hypes["_qcar_amp_requested"] = not opt.no_amp

    if opt.batch_size < 1 or opt.accumulation_steps < 1:
        raise ValueError("batch_size and accumulation_steps must both be positive")

    print("Dataset building (train + validate; test/ is NEVER touched here)")
    train_ds = build_dataset(hypes, visualize=False, train=True)
    val_ds = build_dataset(hypes, visualize=False, train=False)
    print("  train: %d frames   validate: %d frames" % (len(train_ds), len(val_ds)))

    # A validation set with no visible target produces no validation losses,
    # never saves a best checkpoint, and makes the final evaluation command
    # fail later for a much less obvious reason.  Check this before allocating
    # a model or starting a long GPU run.  The held-out test split remains
    # untouched.
    val_positive_frames = sum(
        int(val_ds[index]["ego"]["object_bbx_mask"].sum()) > 0
        for index in range(len(val_ds))
    )
    print("  validate positive frames: %d/%d" %
          (val_positive_frames, len(val_ds)))
    if val_positive_frames == 0:
        raise RuntimeError(
            "Validation split contains no front-visible target frames. "
            "Fine-tuning cannot select a best checkpoint. Collect another "
            "trajectory or create a documented development split from train; "
            "do not claim adjacent same-trajectory frames as an independent test."
        )
    if opt.preflight_only:
        print("Preflight PASS; model/GPU setup intentionally skipped.")
        return

    train_loader = DataLoader(train_ds, batch_size=opt.batch_size, num_workers=2,
                              collate_fn=train_ds.collate_batch_train, shuffle=True,
                              pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=opt.batch_size, num_workers=2,
                            collate_fn=train_ds.collate_batch_train, shuffle=False,
                            pin_memory=True, drop_last=False)

    print("Creating model / loss / optimizer")
    model = train_utils.create_model(hypes)

    # Freeze the earliest N EfficientNet blocks of the image encoder, if
    # requested. Generic edge/texture features live in the early blocks;
    # freezing them reduces the parameters free to overfit the 86 training
    # scenarios while still letting the later blocks (plus everything
    # downstream: LSS depth net, backbone, heads) adapt to QCar appearance.
    # Frozen params simply get no gradient -- Adam skips them at step time,
    # no optimizer-side filtering needed.
    freeze_n = hypes.get("_qcar_freeze_encoder_blocks", 0)
    if freeze_n > 0:
        trunk = model.encoder_m2.camencode.trunk
        for param in trunk._conv_stem.parameters():
            param.requires_grad = False
        for param in trunk._bn0.parameters():
            param.requires_grad = False
        for block in trunk._blocks[:freeze_n]:
            for param in block.parameters():
                param.requires_grad = False
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Froze stem + first %d/%d EfficientNet blocks: "
              "%d params frozen, %d params trainable" %
              (freeze_n, len(trunk._blocks), frozen, trainable))

    criterion = train_utils.create_loss(hypes)
    optimizer = train_utils.setup_optimizer(hypes, model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(torch.cuda.is_available() and not opt.no_amp)
    hypes["_qcar_amp_enabled"] = use_amp
    hypes["_qcar_effective_batch_size"] = (
        opt.batch_size * opt.accumulation_steps
    )

    resume_state = None
    if opt.model_dir:
        saved_path = opt.model_dir
        state_path = os.path.join(saved_path, "training_state_last.pth")
        if os.path.isfile(state_path):
            resume_state = torch.load(state_path, map_location="cpu")
            model.load_state_dict(resume_state["model_state_dict"], strict=True)
            optimizer.load_state_dict(resume_state["optimizer_state_dict"])
            init_epoch = int(resume_state["next_epoch"])
            scheduler = train_utils.setup_lr_schedular(hypes, optimizer,
                                                        init_epoch=init_epoch)
            if "scheduler_state_dict" in resume_state:
                scheduler.load_state_dict(resume_state["scheduler_state_dict"])
            print("resuming full training state from epoch %d" % init_epoch)
        else:
            init_epoch, model = train_utils.load_saved_model(saved_path, model)
            scheduler = train_utils.setup_lr_schedular(hypes, optimizer,
                                                        init_epoch=init_epoch)
            print("resuming weights only from epoch %d" % init_epoch)
    else:
        init_epoch = 0
        saved_path = train_utils.setup_train(hypes)
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer)

        # our own scaffolding key (not a stock HEAL field): a yaml built for an
        # architecture we have no trained weights for yet (e.g. Pyramid) sets
        # this to null on purpose. Warn loudly rather than silently training
        # heter_pyramid_collab from random init on 90 frames and letting that
        # look like a normal fine-tuning run in the logs.
        pretrained = hypes.get("_qcar_pretrained_checkpoint")
        encoder_pretrained = hypes.get(
            "_qcar_encoder_initialization_checkpoint"
        )
        if "_qcar_pretrained_checkpoint" in hypes and not pretrained:
            print("\n" + "!" * 78)
            print("! _qcar_pretrained_checkpoint is null -- no pretrained weights exist")
            print("! for the complete architecture.")
            if encoder_pretrained:
                repository_root = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "../..")
                )
                encoder_path = (encoder_pretrained if os.path.isabs(encoder_pretrained)
                                else os.path.join(repository_root,
                                                  encoder_pretrained))
                if not os.path.isfile(encoder_path):
                    raise FileNotFoundError(encoder_path)
                encoder_checkpoint = torch.load(encoder_path,
                                                map_location="cpu")
                encoder_state = encoder_checkpoint.get(
                    "model_state_dict", encoder_checkpoint
                )
                prefixes = tuple(hypes.get(
                    "_qcar_encoder_initialization_prefixes",
                    ["encoder_m2."],
                ))
                current_state = model.state_dict()
                compatible_encoder = {
                    key: value for key, value in encoder_state.items()
                    if key.startswith(prefixes) and key in current_state
                    and tuple(value.shape) == tuple(current_state[key].shape)
                }
                if not compatible_encoder:
                    raise RuntimeError(
                        "No compatible encoder tensors found in %s" %
                        encoder_path
                    )
                model.load_state_dict(compatible_encoder, strict=False)
                print("! transferred %d compatible encoder tensors from:" %
                      len(compatible_encoder))
                print("! %s" % encoder_path)
                print("! all non-encoder Pyramid modules remain randomly initialized.")
            else:
                print("! Training FROM RANDOM INITIALISATION on %d frames." %
                      len(train_ds))
            print("! This is not complete-model fine-tuning; treat any resulting")
            print("! number as experimental until a real checkpoint is available.")
            print("!" * 78 + "\n")
        elif pretrained:
            print("loading pretrained weights before fine-tuning:", pretrained)
            ckpt = torch.load(pretrained, map_location="cpu")
            incompatible = model.load_state_dict(
                ckpt.get("model_state_dict", ckpt), strict=False)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError(
                    "Pretrained checkpoint is not architecture-compatible: "
                    "missing=%s unexpected=%s" %
                    (incompatible.missing_keys, incompatible.unexpected_keys)
                )

    # dump the FULLY resolved config next to the checkpoints -- reproducibility,
    # not a verbal description of what settings were used.
    with open(os.path.join(saved_path, "resolved_hypes.json"), "w") as f:
        json.dump(hypes, f, indent=2, default=str)

    model.to(device)
    # Optimizer state restored from a CPU checkpoint must follow the model.
    for optimizer_state in optimizer.state.values():
        for key, value in optimizer_state.items():
            if torch.is_tensor(value):
                optimizer_state[key] = value.to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if resume_state is not None and "scaler_state_dict" in resume_state:
        scaler.load_state_dict(resume_state["scaler_state_dict"])
    print("Training precision: %s; effective batch size: %d" %
          ("AMP" if use_amp else "FP32",
           opt.batch_size * opt.accumulation_steps))
    # HeterPyramidCollab consumes this setting in its loss contract but does
    # not expose it as a model attribute in this HEAL checkout.  Read the
    # source-of-truth YAML so the pyramid occupancy loss is not silently lost.
    supervise_single = bool(
        hypes["model"]["args"].get("supervise_single", False)
    )
    single_weight = hypes["train_params"].get("single_weight", 1.0)
    epoches = hypes["train_params"]["epoches"]
    lowest_val_loss = (resume_state.get("best_val_loss", 1e5)
                       if resume_state is not None else 1e5)
    lowest_val_epoch = (resume_state.get("best_val_epoch", -1)
                        if resume_state is not None else -1)
    run_log = (resume_state.get("run_log", [])
               if resume_state is not None else [])

    for epoch in range(init_epoch, max(epoches, init_epoch)):
        model.train()
        t0 = time.time()
        processed, missing = 0, 0
        train_losses = []
        accumulated = 0
        optimizer.zero_grad(set_to_none=True)
        for batch_data in train_loader:
            if batch_data is None:
                missing += 1
                continue
            processed += 1
            batch_data = train_utils.to_device(batch_data, device)
            batch_data["ego"]["epoch"] = epoch
            with torch.cuda.amp.autocast(enabled=use_amp):
                output_dict = model(batch_data["ego"])
                loss = criterion(output_dict, batch_data["ego"]["label_dict"])
                if supervise_single:
                    loss = loss + single_weight * criterion(
                        output_dict,
                        batch_data["ego"]["label_dict_single"],
                        suffix="_single",
                    )
            scaler.scale(loss / opt.accumulation_steps).backward()
            accumulated += 1
            if accumulated == opt.accumulation_steps:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.gradient_clip)
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)
                accumulated = 0
            train_losses.append(loss.item())

        if accumulated:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.gradient_clip)
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(set_to_none=True)

        print("epoch %d: processed %d/%d steps (%.0f%%), train loss %.4f, %.1fs" % (
            epoch, processed, processed + missing,
            100.0 * processed / max(processed + missing, 1),
            statistics.mean(train_losses) if train_losses else float("nan"),
            time.time() - t0))

        val_losses = []
        model.eval()
        with torch.inference_mode():
            for batch_data in val_loader:
                if batch_data is None:
                    continue
                batch_data = train_utils.to_device(batch_data, device)
                batch_data["ego"]["epoch"] = epoch
                # Validation always runs in fp32: AMP autocast has no
                # GradScaler here (that only guards backward/optimizer.step),
                # so a validate-only forward pass can silently overflow fp16
                # and produce NaN as the weights evolve -- confirmed by
                # find_nan_frame.py finding 0/1952 bad frames at epoch-1
                # weights, i.e. the data is fine, this was a precision
                # artifact of validating under autocast.
                with torch.cuda.amp.autocast(enabled=False):
                    output_dict = model(batch_data["ego"])
                    val_loss = criterion(output_dict, batch_data["ego"]["label_dict"])
                    if supervise_single:
                        val_loss = val_loss + single_weight * criterion(
                            output_dict,
                            batch_data["ego"]["label_dict_single"],
                            suffix="_single",
                        )
                val_losses.append(val_loss.item())

        val_loss = statistics.mean(val_losses) if val_losses else float("nan")
        print("  validate loss: %.4f (%d/%d batches processed; %d positive frames)" %
              (val_loss, len(val_losses), len(val_loader), val_positive_frames))
        run_log.append({"epoch": epoch, "processed_steps": processed,
                        "missing_steps": missing,
                        "train_loss": statistics.mean(train_losses) if train_losses else None,
                        "val_loss": val_loss if val_losses else None})

        if val_losses and val_loss < lowest_val_loss:
            lowest_val_loss = val_loss
            ckpt = os.path.join(saved_path, "net_epoch_bestval_at%d.pth" % (epoch + 1))
            for old in glob.glob(os.path.join(saved_path,
                                              "net_epoch_bestval_at*.pth")):
                if old != ckpt:
                    os.remove(old)
            torch.save(model.state_dict(), ckpt)
            lowest_val_epoch = epoch + 1
            print("  new best val checkpoint: %s" % ckpt)

        scheduler.step(epoch)
        torch.save({
            "next_epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_loss": lowest_val_loss,
            "best_val_epoch": lowest_val_epoch,
            "run_log": run_log,
        }, os.path.join(saved_path, "training_state_last.pth"))
        train_ds.reinitialize()

    with open(os.path.join(saved_path, "run_log.json"), "w") as f:
        json.dump(run_log, f, indent=2)
    print("\nTraining finished. Checkpoints + resolved_hypes.json + run_log.json in:", saved_path)
    if hypes.get("test_dir"):
        print("Run eval_qcar.py --model_dir %s --split test ONLY ONCE, at the very end." % saved_path)
    else:
        print("No same-trajectory test is configured. Final evaluation requires a new independent QCar trajectory.")


if __name__ == "__main__":
    main()
