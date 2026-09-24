# QCar dataset-building pipeline — how to run it, and every flag it takes

For *what* each dataset is and its reproducibility status, see `CONVERSION_NOTES.md`.
For a walkthrough of every script's internal logic (functions, why it's built
that way), see `14 GUIA` in the project's research notes (kept outside this repo; ask the maintainer). **This file is the
practical "how do I run this" reference — every flag, what it defaults to,
and why**, so someone new to the project (or a fresh clone on a different
machine) can run the pipeline without editing any script.

## Flag convention

Every script under `scripts/` that reads or writes a path takes that path as
a `--flag`, never hardcoded. Two rules make this consistent everywhere:

1. **Every default lives in the folder's `conf.json`, never in the script.**
   The scripts hardcode no path or value: with no flags they use exactly what
   `conf.json` says, and they stop with a clear error if a required key is
   missing. Flags are optional, one-run overrides for anyone who doesn't want
   to edit the file. Relative paths in `conf.json` are resolved against the
   `qcar_testbed_integration/` root, so the file works on any clone;
   machine-specific locations (the external pretrain drive) go in as
   absolute paths and are edited per machine.
2. **Scripts that are also imported as libraries** (`build_coop_train_val_dataset.py`
   is imported by every `diagnose_*.py` script) build their flag parser with
   `argparse.ArgumentParser(add_help=False)` and call `parse_known_args()`,
   not `parse_args()`. This means importing the module never crashes on an
   unrelated flag the *importing* script defines, and never swallows that
   script's own `-h`/`--help`. Because `add_help=False` also means `-h`
   wouldn't otherwise be caught by this module's own parser when run
   directly, each such script adds one explicit guard right before parsing:
   ```python
   if __name__ == "__main__" and any(a in ("-h", "--help") for a in sys.argv[1:]):
       _parser.print_help()
       raise SystemExit(0)
   ```
   **This guard is load-bearing, not decorative** — `build_coop_train_val_dataset.py`
   and the three `build_pretrain_*.py` scripts run `shutil.rmtree()` on their
   output directory unconditionally in `main()`, guarded only by
   `if __name__ == "__main__"`. Before this guard existed, `--help` on any of
   them silently fell through to a full destructive rebuild instead of
   printing usage (caught 2026-09-18 after it happened for real — see
   `CONVERSION_NOTES.md`'s history section). If you add a new flag-taking
   script here, copy this pattern, don't skip it.

Adopt the same two rules for any new script you add to this pipeline.

## `scripts/convert/bag_to_dataset_rosbags.py`

Every default comes from `scripts/convert/conf.json`; each flag below
overrides one key for one run. The shipped defaults point at trajectory 00
of the `.198` car (`data/qcar_onboard/Training/ExperimentNo1/192.168.1.198/`);
since that output already exists and `overwrite`/`add` are `false`, running
with no flags refuses to touch it -- pass `--bag`/`--output` for a new bag.

| flag | conf.json key (current value) | meaning |
|---|---|---|
| `--bag` | `bag` (trajectory 00 of `.198`) | path to the raw `.bag` |
| `--output` | `output` (its `converted_*_00/`) | output directory for the per-frame folders |
| `--overwrite` / `--add` | `overwrite` / `add` (`false`) | delete, or append to, an existing output |
| `--fps` | `fps` (10.0) | reference sampling rate |
| `--wait-sec` | `wait_sec` (0.05) | bounded-wait window before deciding a frame |
| `--camera-tolerance` | `camera_tolerance` (0.04) | max time gap to match a non-reference camera |
| `--vicon-max-age` | `vicon_max_age` (0.10) | max age of the Vicon reading used |
| `--vicon-pose-key` | `vicon_pose_key` (`auto`) | `auto`/`ego`/`target` |
| `--trim-start-sec`, `--duration-sec`, `--trim-to-motion`, `--motion-threshold-m`, `--motion-confirm-samples` | same names (`duration_sec: null` = whole bag) | trimming/motion-detection controls |
| `--png-compression-level` | `png_compression_level` (0) | 0-9 |

## `scripts/build/` — reorganized 2026-09-23 into 3 subfolders

Each subfolder has its own `conf.json`, the single source of every default
below — edit that file instead of typing flags every run. See
`scripts/build/SCRIPTS.md` for a one-line purpose per script.

### `scripts/build/cooperative/` — the active, real pipeline

**`build_coop_train_val_dataset.py`** — builds `CoopFront/`.

| flag | `cooperative/conf.json` key (current value) | meaning |
|---|---|---|
| `--calib-root` | `calib_root` (`calibration/calibration_Matries`) | calibration tree root |
| `--onboard-root` | `onboard_root` (`data/qcar_onboard/Training/ExperimentNo1`) | root holding `<ip>/dataset/` per car |
| `--out-dir` | `out_dir` (`data/qcar_dataset/pipeline/datasets/CoopFront`) | where to write the dataset |
| `--vis-masks-dir` | `vis_masks_dir` (`.../pipeline/shared/visibility_masks`) | per-car BEV visibility masks |
| `--walls-json` | `walls_json` (`.../pipeline/shared/walls.json`) | real wall geometry for occlusion |
| `--val-indices` | `val_indices` (`""` = all train) | comma-separated trajectory indices held out for validate |
| `--agents` | `agents` (`1,2`) | comma-separated agent ids to include |
| *(no flag)* | `pair_tolerance_sec` (`0.04`) | wall-time pairing tolerance |
| *(no flag)* | `static_targets` | fixed Vicon poses of the parked cars |
| `--max-trajectories` | `max_trajectories` (`null` = all) | cap to the first N trajectories -- a fast, real-data smoke-test build |

**Destroys and rebuilds `--out-dir` every run** (`shutil.rmtree` then
regenerate) — it is meant to be idempotent given the same inputs, not
incremental.

**`build_inference.py`** — same real captures, no ground truth, builds
`InferenceFront/`. Same `--calib-root`/`--onboard-root`/`--vis-masks-dir` flags and conf
keys; `--out-dir` comes from `inference_out_dir`.

### `scripts/build/pretrain_encoder/` — blocked on an external drive

`build_pretrain_manifest.py` → `build_pretrain_labels.py` →
`build_pretrain_dataset.py`, run in that order, each consuming the previous
one's output. All three share `pretrain_encoder/conf.json`: `src_root` (external media,
absolute path, edit per machine), `calib_root`, `out_dir`, `manifest_json`,
`labels_json`, `ego_vis_mask` — each with a matching `--flag`.
`build_pretrain_dataset.py` also destroys and rebuilds its `--out-dir`; the
other two just write one JSON file each (`manifest_json`, `labels_json`), no
rmtree.

### `scripts/build/tools/`

**`build_visibility_mask.py`** — dry-run by default; `--overwrite` to
actually write. Defaults come from `tools/conf.json` (`calib_root`, `out_dir`,
`cars`, `overwrite`, `coop_yaml`).

**`build_smoketest_dev_split.py`** — re-splits the already-built
`datasets/Smokes/SmokeTestFront/` into `datasets/Smokes/SmokeTestFront_dev/`.
`--source`/`--destination` default to `tools/conf.json`'s `smoketest_source`/
`smoketest_destination` (scenario and cav ids are `smoketest_*` keys too).
Refuses to overwrite an existing destination.

## `scripts/diagnose/*.py` — all take `--model_dir` (required) and `--no_amp`

Every diagnostic script needs a trained run to inspect:

| flag | meaning |
|---|---|
| `--model_dir` | path to a run under `opencood/logs/<run>/` (required) |
| `--no_amp` | disable mixed-precision inference |

The `diagnose_*.py` scripts import `build_coop_train_val_dataset.py`, so
their CoopFront settings (paths, walls, static targets) come from
`scripts/build/cooperative/conf.json`.

`find_nan_frame.py` is the exception (a `main()`-only script, no `--model_dir`);
its defaults live in `scripts/diagnose/conf.json`:

| flag | conf.json key (current value) | meaning |
|---|---|---|
| `--heal-root` | `heal_root` (`..`, i.e. HEAL/) | HEAL repo root, so `opencood/hypes_yaml/...` resolves |
| `--hypes` | `hypes` (`qcar/configs/camera_attfuse_pretrain.yaml`) | relative to `--heal-root` |
| `--checkpoint-glob` | `checkpoint_glob` (`opencood/logs/HeterBaseline_opv2v_camera_attfuse_qcar_pretrain_*`) | relative to `--heal-root`; last match (sorted) is used |
| `--checkpoint-name` | `checkpoint_name` (`net_epoch_bestval_at1.pth`) | checkpoint file inside that run |

## `scripts/visualize/*.py`

`visualize_boxes.py` takes its inputs as positional/required CLI args already
(frame dir, output dir) — no hardcoded paths to begin with.

`analyze_bev_visibility_mask.py` — defaults in `scripts/visualize/conf.json`:

| flag | conf.json key (current value) |
|---|---|
| `--calib-root` | `calib_root` (`calibration/calibration_Matries`) |
| `--masks-root` | `masks_root` (`data/qcar_dataset/pipeline/shared/visibility_masks`) |
| `--cars` | `cars` (`52775,52776`) |

## `opencood` itself must be importable

None of these scripts add the HEAL repo root to `sys.path` — they rely on
`opencood` being pip-installed in editable mode (`python setup.py develop`
from the HEAL repo root, the first-run setup step in the root
`README.md`'s Installation section). If `import opencood` fails, that setup step wasn't
run, not a bug in these scripts.
