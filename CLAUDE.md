# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

HEAL (ICLR 2024) — an extensible framework for open heterogeneous collaborative perception, built on top of the
OpenCOOD codebase (the installed package is still named `OpenCOOD`; the import root is `opencood`). It supports
LiDAR-only, camera-only, and heterogeneous multi-agent 3D detection across OPV2V, V2XSet, V2X-Sim 2.0 and DAIR-V2X-C.

There is **no test suite, linter, or CI** in this repo. Verification means running training/inference on a config.

## Environment & build

```bash
conda activate heal                  # python 3.8, pytorch 1.12 + cudatoolkit 11.6
pip install -r requirements.txt
python setup.py develop              # editable install of `opencood`
pip install spconv-cu116             # or spconv 1.2.1 — see below
python opencood/utils/setup.py build_ext --inplace       # bbox IoU / NMS cython ext (required)
python opencood/pcdet_utils/setup.py build_ext --inplace # only for FPV-RCNN
```

spconv version matters: the released HuggingFace checkpoints were saved under **spconv 1.2.1** and have a wrong
input-channel count for SECOND-based models. They load under 1.2.1 (no sanity check) but not cleanly under 2.x.
Do not build new work on those checkpoints.

First-run setup of the agent-type assignment files:

```bash
mkdir -p opencood/logs
cp -r opencood/modality_assign opencood/logs/heter_modality_assign
```

## Core commands

```bash
# train (fresh)
python opencood/tools/train.py -y opencood/hypes_yaml/opv2v/LiDAROnly/lidar_fcooper.yaml
# train (resume / train from a prepared log dir — see the -y None idiom below)
python opencood/tools/train.py -y None --model_dir opencood/logs/<run>
# DDP
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --use_env \
  opencood/tools/train_ddp.py -y None --model_dir opencood/logs/<run>

# inference
python opencood/tools/inference.py --model_dir opencood/logs/<run> [--fusion_method intermediate]
# open-heterogeneous eval: adds m1..m4 into the scene incrementally, overriding
# mapping_dict / comm_range in the run's config.yaml
python opencood/tools/inference_heter_in_order.py --model_dir opencood/logs/<run>

# merge stage-2 aligned agent models onto the stage-1 collaboration base
# (base dir second-to-last, output dir last; best checkpoints are auto-selected)
python opencood/tools/heal_tools.py merge_final <m2_dir> <m3_dir> <m4_dir> <m1_base_dir> <out_dir>
```

**The `-y None --model_dir` idiom is central.** When `--model_dir` is given, `load_yaml` discards `hypes_yaml`
entirely and reads `<model_dir>/config.yaml` instead (`opencood/hypes_yaml/yaml_utils.py:29`). The HEAL workflow is
therefore: `mkdir` a log dir, `cp` the chosen hypes yaml into it as `config.yaml`, then train with `-y None`.

## Architecture

Everything is **config-driven through string→class resolution**. A yaml names a `core_method`; the framework
imports or looks up the matching class. There is no explicit registration decorator, so *renaming a class or file
silently breaks configs*.

Five resolution points, each with different rules:

| What | Where | Rule |
|---|---|---|
| Model | `tools/train_utils.py:create_model` | `hypes.model.core_method` → module `opencood.models.<name>`, class whose lowercase name equals `<name>` with `_` stripped |
| Loss | `tools/train_utils.py:create_loss` | same scheme under `opencood.loss.` |
| Dataset | `data_utils/datasets/__init__.py:build_dataset` | `fusion.core_method` + `fusion.dataset` composed into a factory call (see below) |
| Pre/post processor | `data_utils/{pre,post}_processor/__init__.py` | explicit `__all__` dict lookup |
| Heterogeneous encoder | `models/heter_pyramid_collab.py` and siblings | per-modality `core_method` → class in `opencood/models/heter_encoders.py` |

### Dataset construction is a mixin factory, not a class hierarchy

`build_dataset` composes a *fusion strategy* with a *dataset source*:

```python
getIntermediateheterFusionDataset(OPV2VBaseDataset)(params, visualize, train)
```

Each `get<Fusion>FusionDataset(base_cls)` returns a freshly-defined class subclassing the given base. So fusion
files (`intermediate_heter_fusion_dataset.py`, `late_fusion_dataset.py`, …) hold the collaboration logic, and
`basedataset/*.py` hold the per-dataset file walking and parsing. Adding a dataset means adding a base class plus
its name to the asserts and imports in `datasets/__init__.py`.

### Yaml is preprocessed before anyone sees it

`load_yaml` evaluates the config's own `yaml_parser` key (`load_general_params`, `load_point_pillar_params`,
`load_lift_splat_shoot_params`, …) which *derives* anchor boxes, grid sizes and feature resolutions from the lidar
range and voxel size. Changing `cav_lidar_range` or `voxel_size` therefore silently changes tensor shapes
downstream. Anchors and `grid_size` in a config are computed, not authored.

### The heterogeneity system (`m1`, `m2`, … identifiers)

Two independent uses of the same identifiers, easy to conflate:

1. **`opencood/modality_assign/*.json`** (copied to `opencood/logs/heter_modality_assign/`) assigns an agent type
   to each CAV in each scene so validation scenarios are fixed and comparable across methods. Generated by
   `assign_modality_4*` in `opencood/utils/heter_utils.py`.
2. **`heter.modality_setting` in a method yaml** defines what `m1`/`m2` actually *are* — sensor type, encoder
   `core_method`, preprocessing, backbone, aligner.

`heter.mapping_dict` bridges the two: it maps the json's agent type onto the types this experiment defines (a
homogeneous camera experiment maps everything to `m2`). **`mapping_dict` is ignored during training** — agents are
randomly assigned any type present in the yaml, as data augmentation — and only takes effect at inference.
`opencood/utils/heter_utils.py:Adaptor` implements this, driven by `heter.ego_modality` (e.g. `"m1"` or `"m1&m2"`).

Models build per-modality submodules by `setattr(self, f"encoder_{modality_name}", …)`, so checkpoint keys carry
prefixes like `encoder_m1.` / `backbone_m1.`. Old CoAlign-style configs without `m*` identifiers still train here
and produce unprefixed keys for the same architecture; `rename_model_dict_keys` in `opencood/utils/model_utils.py`
converts between the two.

### HEAL's two-stage training

Stage 1 trains a collaboration base (LiDAR + PointPillars + Pyramid Fusion). Stage 2 trains each new agent type
*alone*, initialized from the stage-1 checkpoint with the Pyramid Fusion parameters frozen, so new types align to
the base's feature space without retraining it. Stage-2 runs are independent and parallelizable. `heal_tools.py
merge_final` then stitches the per-type encoders onto the base for a single collaborative inference model.

## QCar real-testbed integration (separate branch)

`main` is kept as a clean diff against `upstream` (yifanlu0227/HEAL) on purpose, so it can be rebased/synced and any
eventual contribution has an honest history. The Concordia QCar physical-testbed integration — camera_attfuse
configs for real sensor data, a depth-balanced BCE loss, and train/eval/visualize patches for driving HEAL on live
QCar CSI camera streams — lives on the `qcar-testbed-integration` branch instead of `main`, so it can mature toward
a PR without carrying unrelated commits.

Everything that isn't code (raw QCar recordings, calibration reports, exploratory notebooks, checkpoints/tensors
used only for the physical demo) has been moved out of this repo entirely, into the sibling project
`../HEAL-Concordia/`. See `../HEAL-Concordia/CLAUDE.md` for what's there and how it relates to this branch, and the
Obsidian vault's `concordia/controller-project/` notes for the research context (session logs, roadmap, defense
priorities) driving this work.
