# QCar Real-Testbed Integration

> Developed by **Kevin Saavedra** (MSc, University of Calabria)
> Collaborators **Daryel León** (PhD student, Concordia University)
	     **Wissam Fawaz** (Lebanese American University). 
> Supervisors : **Chadi Assi** (Host, Concordia University) 
> 	     **Floriano De Rango** (Home, University of Calabria), with
> 
> As a Part of the MITACS program.

Camera-only HEAL (`camera_attfuse`, and one Pyramid Fusion config) driven by
live sensor data from Quanser QCar physical vehicles, instead of the
simulated OPV2V/V2XSet/V2X-Sim/DAIR-V2X-C datasets. This directory holds
everything that is specific to that real-sensor pipeline; nothing under
`opencood/` outside this directory and `opencood/loss/` is modified.

## Why a separate directory instead of editing HEAL's own tools

Each script here is a thin wrapper around HEAL's existing machinery
(`create_model`, `create_loss`, `build_dataset`, `train_utils`, HEAL's own
decode+NMS path) rather than a reimplementation, kept out of
`opencood/tools/` so the two behavioral differences that real QCar sensor
data actually requires — see `patch_1cam_loader.py` and
`patch_real_extrinsic.py` — are opt-in monkeypatches, applied only when a
script here explicitly imports them. Stock HEAL configs and the simulated
datasets are completely unaffected. Each file's own docstring explains the
specific "why" in full; this README is only the map.

| File | Role |
|---|---|
| `train_qcar.py` | Training entry point for QCar runs (pretrain / coop / onlyfront). Mirrors `opencood/tools/train.py` but skips `open3d`-dependent visualization and keeps negative-only batches. |
| `eval_qcar.py` | Lean inference/eval (decode + NMS + recall@IoU), no `open3d`, for validating a QCar checkpoint. |
| `visualize_inference.py` | Draws predictions vs. ground truth on the camera image for a handful of frames. |
| `patch_1cam_loader.py` | Monkeypatches `OPV2VBaseDataset.find_camera_files` to read only `camera0` (front), since the QCar rig is single-camera and HEAL's loader hardcodes 4 CARLA cameras. |
| `patch_real_extrinsic.py` | Monkeypatches `OPV2VBaseDataset.get_ext_int` to use the QCar rig's real camera-to-vehicle rotation instead of HEAL's CARLA/UE4-to-OpenCV fix-up matrix, which is wrong for non-simulated poses. |
| `../loss/point_pillar_depth_balanced_bce_loss.py` | Balanced-BCE variant of `PointPillarDepthLoss` for the small/imbalanced QCar detection task; `SigmoidFocalLoss` collapsed to all-negative here. |

## Configs

`opencood/hypes_yaml/opv2v/CameraOnly/qcar_real/`:

- `camera_attfuse_pretrain.yaml` — single-agent encoder pretraining.
- `camera_attfuse_coop.yaml` — two-agent cooperative fine-tune from the pretrain checkpoint.
- `camera_attfuse_onlyfront.yaml` — single-agent, front-camera-only fine-tune.
- `camera_attfuse_qcar.yaml` — the general QCar camera_attfuse config.
- `camera_pyramid_onlyfront.yaml` — Pyramid Fusion variant of the front-only setup.

`_qcar_*`-prefixed keys in these yamls (e.g. `_qcar_pretrained_checkpoint`,
`_qcar_freeze_encoder_blocks`) are this project's own scaffolding, not stock
HEAL fields; `train_qcar.py` reads them and warns loudly rather than
training silently from random init when a checkpoint path is left
unfilled.

## Dataset layout expected by these configs

`root_dir` / `validate_dir` / `test_dir` in the configs above point at
`qcar_dataset/pipeline/datasets/<split-name>/{train,validate,test}`, relative to the
repo root, in the same per-frame layout HEAL's own OPV2V loader expects
(`<frame_id>_camera0.png` + `<frame_id>.yaml`). That path is a symlink
(`qcar_dataset -> HEAL-Concordia/data/qcar_dataset`) into `HEAL-Concordia/`,
a `.gitignore`d directory nested at the repo root holding the QCar
recordings, calibration reports, and conversion tooling that produce this
layout — data only, no framework code, and not part of this repository's
git history or of the upstream-facing diff.

## Usage

```bash
python opencood/qcar_patches/train_qcar.py -y opencood/hypes_yaml/opv2v/CameraOnly/qcar_real/camera_attfuse_pretrain.yaml
python opencood/qcar_patches/eval_qcar.py --model_dir opencood/logs/<run> --split validate
python opencood/qcar_patches/visualize_inference.py --model_dir opencood/logs/<run> --split validate --n 8 --out <dir>
```
