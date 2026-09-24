# Every script in this repo — what it does, one place

Three areas, in the order a new recording flows through them: **getting data
off the cars** (`data/qcar_onboard/_tools/`), the **dataset-building pipeline**
(`data/qcar_dataset/pipeline/scripts/`), and the **calibration tools**
(`calibration/calibration_Matries/tools/`). Training, evaluation and the model
zoo are not here: they live in the HEAL repo's `qcar/` package (start at
`qcar/README.md`). This file is the single index;
deeper detail (every flag, internal function-by-function logic) lives in the
files linked from each section — this page never duplicates that, only
summarizes and points there.

**Raw data is not in git.** The ROS bags, per-frame captures and built
datasets (~27 GB) are kept by the lab, outside the repository. Ask the
maintainer for access, and place them under `data/` as the READMEs below
describe.

---

## 0. Getting data off the cars — `data/qcar_onboard/_tools/`

Details: `data/qcar_onboard/README.md`.

| script | what it does |
|---|---|
| `pull_qcar_bags.py` | Copies a car's recorded `.bag` files to this workstation over SSH (read-only on the car: probes plus a `tar` stream) and writes a `BAGS_MANIFEST.md` with the car's identity and the sha256 of what was pulled. Usage: `python3 pull_qcar_bags.py <car-ip>`; defaults in `_tools/conf.json`. |
| `cross_check_vicon_labeling.py` | Checks whether the Vicon ego/target labeling bug documented in `bag_to_dataset_rosbags.py` affected the 9 cooperative trajectories (it did not, verified 2026-09-18). `--diagnostic` prints the raw distance table. |

---

## 1. Dataset pipeline — `data/qcar_dataset/pipeline/scripts/`

Full flag reference: `data/qcar_dataset/pipeline/README.md`.
Full internal-logic walkthrough: `14 GUIA` in the project's research notes (kept outside this repo; ask the maintainer).
Dataset-level reproducibility status: `data/qcar_dataset/pipeline/CONVERSION_NOTES.md`.

### `scripts/convert/`

| script | what it does |
|---|---|
| `bag_to_dataset_rosbags.py` | Splits one raw ROS1 `.bag` (4 cameras + Vicon, mixed by arrival time) into per-frame folders (`front/back/left/right.png` + `ego_vicon_pose.json` + `timestamp.json`), using a causal bounded-wait sync so no "future" data leaks into a decision. The only real entry point of the whole pipeline — everything else consumes its output. |

### `scripts/build/` — reorganized 2026-09-23 into 3 subfolders, each with its own `conf.json`

Full detail: `scripts/build/SCRIPTS.md`. `build_smoketest_dataset.py` and
`build_train_val_dataset.py` (both historical, broken on purpose, raw
inputs deleted 2026-09-18) were deleted the same day — use
`build_coop_train_val_dataset.py --max-trajectories N` for a fast
real-data/real-code smoke-test build instead.

| script | status | what it does |
|---|---|---|
| `cooperative/build_coop_train_val_dataset.py` | ✅ active | Builds `CoopFront/` — the main N-agent (currently 2) cooperative dataset (9 real trajectory pairs, 4-vehicle scenes with 2 static wall-occluded targets, real `wall_geometry.py` occlusion check). Destructive: rebuilds its output dir from scratch every run. `--max-trajectories N` caps it to a fast smoke-test subset. |
| `cooperative/build_inference.py` | ✅ active | Same real captures, no ground truth, builds `InferenceFront/` for pure deployment/inference. |
| `cooperative/wall_geometry.py` | ✅ active | Real wall-segment occlusion geometry (`shared/walls.json`) — the only wall check the cooperative scripts use. |
| `cooperative/Report_creation.py` | ✅ active | Reads `build_coop_train_val_dataset.py`'s own functions directly (no separate recomputation) to write a human-readable Markdown report per trajectory pair, plus an index, into `CoopFront/reports/` — closest-approach point, per-vehicle visibility, and how many of the 4 vehicles are ever simultaneously visible. |
| `pretrain_encoder/build_pretrain_manifest.py` | ✅ active (blocked on external drive) | Stage 1/3 of the pretrain pipeline: scans the 101-scenario external collision corpus and filters out unusable frames (black warm-up frames, invalid Vicon). Writes `shared/pretrain_manifest.json`. |
| `pretrain_encoder/build_pretrain_labels.py` | ✅ active (blocked on external drive) | Stage 2/3: corrects "Vicon-valid" into "camera-actually-sees-it" by projecting the real 3D box into the real camera and checking it truly lands in frame (a 2026-09-14 audit found 26% of Vicon-valid targets weren't actually in the image). Writes `shared/pretrain_labels.json`. |
| `pretrain_encoder/build_pretrain_dataset.py` | ✅ active (blocked on external drive) | Stage 3/3: writes the final `PretrainFront/` dataset (single-agent, x10 world scale, real front calibration) from the corrected labels. Destructive rebuild of its output dir. |
| `tools/build_visibility_mask.py` | ✅ active | Reconstructs the per-car BEV visibility cone (`shared/visibility_masks/`) from real calibration. Dry-run by default. |
| `tools/build_smoketest_dev_split.py` | ✅ active | Re-splits the already-built `SmokeTestFront/` into `SmokeTestFront_dev/` with a leakage-reduced (purged temporal-block) train/validate split, since the original chronological split put everything in train. |

### `scripts/diagnose/` — post-mortem analysis of an already-trained checkpoint, never build data

| script | what it does |
|---|---|
| `diagnose_coop_breakdown.py` | Breaks down detection performance by category (dynamic vs. cooperation-through-wall) on a trained run. |
| `diagnose_fp_overlap.py` | Cross-checks two independent false-positive filters (real-GT-explained vs. FOV-margin-explained) frame by frame to find the genuinely unexplained residual. |
| `diagnose_fp_scores.py` | Checks whether raising the detection score threshold alone would explain away false positives (finding: no, TP/FP score distributions overlap too much). |
| `diagnose_nms_duplicates.py` | Sweeps `nms_thresh` to find the real tradeoff between merging duplicate boxes on the same target and losing genuine nearby detections. |
| `diagnose_visibility_margin.py` | Tests whether widening the FOV-visibility check's margin reclassifies more frames as "should be visible." |
| `compare_extrinsic_nominal_vs_calibrated.py` | Quantifies how much qcar-52775's real calibrated extrinsic differs from the manufacturer's nominal value — pixel-space projection shift and how many frames flip visibility, over all 9 real trajectories. |
| `find_nan_frame.py` | Scans a validate set one frame at a time (batch size 1, CPU) to find exactly which frame produces NaN/Inf loss. |

### `scripts/visualize/`

| script | what it does |
|---|---|
| `visualize_boxes.py` | Draws predicted/ground-truth 3D boxes projected onto the real camera image, for visual sanity-checking of geometry (especially fisheye edge cases). |
| `analyze_bev_visibility_mask.py` | Reverse-engineers how the BEV visibility mask (its original generator script was lost) was built, by independently recovering each camera's real FOV from its measured K/D and comparing it to the mask's own angular extent. |

### `shared/evidence_through_wall/` — historical, does not run as-is

`show_through_wall.py`, `show_all_three.py`, `show_bev_all_three.py`,
`collage_honest.py` and `diagnose_coop_breakdown.py` produced the 2026-09-14
"seeing through the wall" evidence images (kept locally, not in git). They
predate the path clean-up: they hardcode the original workstation's paths and
import from a `qcar_dataset/Inference` module that no longer exists. They are
kept as a record of how that evidence was made. For new visualizations, use
`scripts/visualize/visualize_boxes.py` or the HEAL repo's `qcar_rounds.ipynb`.

---

## 2. Calibration tools — `calibration/calibration_Matries/tools/`

Results/decisions from these tools: `calibration/calibration_Matries/REPORT.md`.

| script | what it does |
|---|---|
| `calib_web.py` | Live fisheye-calibration assistant, driven from a phone browser pointed at the workstation. Streams the QCar's camera feed and shows, live, which part of the lens' field of view is still unconstrained by checkerboard captures so far — replaces the old workflow of capturing blind and grading afterward. Usage: `python3 calib_web.py --car 192.168.1.198 --camera front`, then open `http://<workstation-ip>:8000` on a phone. Flags: `--car`, `--camera`, `--car-port` (55700), `--http-port` (8000), `--out` (save accepted views), `--cols`/`--rows` (checkerboard, 9x6), `--seed`/`--seed-state` (preload earlier good views). |
| `verify_final_intrinsics.py` | Fits and independently verifies a car's front fisheye intrinsics (K, D) from raw checkerboard photos — full `cv2.fisheye.calibrate` with outlier rejection, checked against 6 quality gates. Car, pattern and every path come from `tools/conf.json`'s `verify_*` keys (flags `--views-dir`/`--initial`/`--new-capture`/`--report`/`--output` override them for one run; the original inputs were in this tool's own directory and are no longer on disk) — this is what got re-run 2026-09-17 against the 1554 raw photos to independently confirm `REPORT.md`'s numbers, not just read them. |
| `build_qcar_pooling_map.py` | Builds a precomputed, static Lift-Splat pooling map (which world points fall into which BEV cell) for the QCar's cameras, in the project's x10-scaled world. **Not currently loaded by any active model code** — a prepared, verified, not-yet-activated optimization path. Needs an external QuantV2X checkout: set the `pool_*` paths in `tools/conf.json`. |
| `build_pooling_groups.py` | Reorganizes a `.bin` from the tool above into a cell-grouped `.bin.groups` file, once, offline. Same "not activated yet" status as above. |
| `car_frame_server.py` | Runs **on the QCar itself** (not the workstation) — keeps one CSI camera open for a whole session and serves fresh JPEG frames over a raw TCP socket on request, avoiding the slow camera-open overhead per capture. Deliberately opens only ONE camera (`QCarCameras.readAll()` would open all four, and one stalled secondary camera can freeze a front-camera calibration session). Usage: `sudo python3 car_frame_server.py --port 55700 --camera front`. Flags: `--port` (55700), `--camera` (front/left/right/back), `--quality` (JPEG quality, 80). |

---

## History

Written 2026-09-18 as a single consolidated index after the dataset-pipeline
scripts were fully de-hardcoded (flags, no absolute paths) and two were
renamed for consistency with `SmokeTestFront` — this file did not exist
before that pass, and the calibration tools had never been described
anywhere outside their own docstrings until now.
