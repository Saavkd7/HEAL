# scripts/build/ — what each file is for

Reorganized 2026-09-23 into 3 subfolders, one `conf.json` per folder (edit
that file's defaults instead of typing flags every run — every flag can
still override it for one run). Updated same day: `build_smoketest_dataset.py`
and `build_train_val_dataset.py` deleted (see "Deleted" at the bottom).

## `cooperative/` — the active, real pipeline

- **`build_coop_train_val_dataset.py`** — the central script. Builds
  `CoopFront/`, the real N-agent (currently 2: qcar-52775/qcar-52776)
  cooperative train/validate dataset, from the real captured `.bag` →
  `converted_*` frames under `qcar_onboard/`. Everything else in this repo's
  active training (`camera_attfuse_coop.yaml`) depends on its output.
  `--max-trajectories N` builds a fast, real-data-and-code smoke-test subset
  instead of the full 9 trajectories.

- **`build_inference.py`** — same real captures, stripped down to exactly
  what HEAL needs to RUN (not train): no ground truth, no `scene_gt`, empty
  `vehicles{}`. Builds `InferenceFront/`, for pure deployment/inference
  where you don't know the answer in advance.

- **`wall_geometry.py`** — real wall-segment occlusion geometry (loads
  `shared/walls.json`, 5 measured segments). Used by
  `build_coop_train_val_dataset.py` (and by `diagnose_coop_breakdown.py`,
  `compare_extrinsic_nominal_vs_calibrated.py` in `scripts/diagnose/`) to
  decide, per real frame, whether a physical wall blocks an agent's line of
  sight to a target. This is the ONLY wall check now — the old fixed
  `WALL_BLOCKED` pair-fact was retired 2026-09-22.

- **`Report_creation.py`** — generates a human-readable Markdown report per
  trajectory (closest approach, visibility histogram, etc.) plus an index,
  by reusing `build_coop_train_val_dataset.py`'s own functions directly — no
  numbers are ever recomputed with separate logic, so a report can't
  silently disagree with the dataset it describes.

## `pretrain_encoder/` — blocked on an external drive (not currently connected)

Three-stage pipeline for `PretrainFront/`, the single-agent dataset that
teaches the encoder to recognize a QCar visually before the cooperation
stage. Run in this order, each consuming the previous one's output. All
three need `/media/saavkd7/C4D26282D2627896/Concordia_QCar_Dataset/Scenarios/`,
which isn't mounted right now — code is real and conf.json-wired, just
untestable until the drive is connected.

- **`build_pretrain_manifest.py`** — filters the 101-scenario collision
  corpus down to usable front-camera frames (drops black warm-up frames,
  invalid Vicon poses); writes `shared/pretrain_manifest.json`.
- **`build_pretrain_labels.py`** — corrects manifest positives: Vicon-valid
  isn't the same as camera-visible. Writes `shared/pretrain_labels.json`.
- **`build_pretrain_dataset.py`** — builds `PretrainFront/` itself from the
  labels, same schema/scale/undistortion conventions as the coop dataset.

## `tools/`

- **`build_visibility_mask.py`** — (re)generates
  `shared/visibility_masks/qcar<NNNNN>_bev_visibility.png`, the per-car BEV
  visibility cone HEAL's own `box_is_visible()` reads at train/inference
  time. Reconstructs it from real calibration (K/D + `cav_lidar_range`),
  since the original generator was lost. Dry-run by default.

- **`build_smoketest_dev_split.py`** — builds a leakage-reduced dev split
  from the still-present `SmokeTestFront/` output (see "Frozen/historical"
  below). Doesn't need any missing raw source, so this one still runs;
  paths come from `tools/conf.json` (`smoketest_*`), refuses to overwrite.

## Frozen / historical (their own raw sources no longer exist as `.bag`s)

`SmokeTestFront/`/`SmokeTestFront_dev/` themselves (the DATA, in
`datasets/`, not a script here) still exist and still train fine — only
*regenerating* `SmokeTestFront/` from scratch is impossible, since its own
builder script was deleted 2026-09-23 (see below) after its raw source
(`opv2v_ready/`, `clean_dataset/`) was removed 2026-09-18. Use
`build_coop_train_val_dataset.py --max-trajectories N` for any future
small/fast dataset need instead.

## Deleted 2026-09-23

- ~~`build_smoketest_dataset.py`~~ — was **BROKEN, DO NOT RUN**: its inputs
  (`opv2v_ready/`, `clean_dataset/`) were removed 2026-09-18. Deleted once
  `build_coop_train_val_dataset.py --max-trajectories` covered the same
  "small dataset for a quick test" need with real, live code. Its documented
  design decisions (marker offset math, x10 scale, same-K undistortion) are
  already duplicated in the live cooperative scripts.
- ~~`build_train_val_dataset.py`~~ — same "BROKEN/DO NOT RUN, source
  deleted 2026-09-18" status, but its own built output was ALSO gone
  (`opv2v_ready/`, never fed a real trained model), so nothing was lost by
  deleting the code — the design notes it held (contiguous chronological
  train/validate/test split for a single continuous trajectory; the same
  fisheye-undistortion technique already live in
  `build_coop_train_val_dataset.py`) are summarized in
  `CONVERSION_NOTES.md`'s History section.
