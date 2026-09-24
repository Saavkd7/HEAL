# QCar -> OPV2V layout: what is real, what is assumed, what is reproducible

Rewritten 2026-09-18, twice — first to describe the current pipeline after an
earlier one (`clean_dataset/` -> `opv2v_layout/`) was removed, then again
after `Inference/` was renamed to `pipeline/` and reorganized by type
(`scripts/{convert,build,diagnose,visualize}/`, `datasets/`, `shared/`). See
"History" below for both changes.

## Layout of this directory

```
pipeline/
  scripts/
    convert/    bag_to_dataset_rosbags.py
    build/      reorganized 2026-09-23 into 3 subfolders, see scripts/build/SCRIPTS.md
                  cooperative/       build_coop_train_val_dataset.py (active), build_inference.py (active),
                                     wall_geometry.py, Report_creation.py, conf.json
                  pretrain_encoder/  build_pretrain_manifest.py / _labels.py / _dataset.py
                                     (blocked on an external drive, see status table below), conf.json
                  tools/             build_visibility_mask.py, build_smoketest_dev_split.py (active), conf.json
    diagnose/   diagnose_*.py, find_nan_frame.py, compare_extrinsic_nominal_vs_calibrated.py
    visualize/  visualize_boxes.py, analyze_bev_visibility_mask.py
  datasets/     CoopFront/, SmokeTestFront/, SmokeTestFront_dev/, PretrainFront/
  shared/       visibility_masks/, walls.json, evidence_through_wall/, pairing_by_walltime.json,
                pretrain_labels.json, pretrain_manifest.json
  CONVERSION_NOTES.md   (this file)
```

`build_smoketest_dataset.py` and `build_train_val_dataset.py` (both
historical, broken on purpose, raw inputs deleted 2026-09-18) were removed
entirely 2026-09-23 — nothing was lost, their design notes are summarized
in "History" below. Use `build_coop_train_val_dataset.py --max-trajectories
N` for a fast real-data smoke-test build instead of either.

Scripts resolve their own paths via `PIPELINE_ROOT` (computed from `__file__`,
walking up out of `scripts/<type>/`, or `scripts/build/<category>/` for the
now-nested build scripts — one `.dirname()`/`.parent` deeper than the other
script types) — never assume a script's own directory holds the data it
reads/writes; check its `PIPELINE_ROOT`/`DST`/`SRC` constants instead.
Cross-script imports (e.g. `diagnose_*.py` importing
`build_coop_train_val_dataset` from `scripts/build/cooperative/`,
`build_pretrain_labels.py` importing `visualize_boxes`) use explicit
`sys.path.insert(0, "...")` — this matters if you ever move a script again.

## Current pipeline (the only one still runnable end to end)

```
qcar_onboard/<ip>/bags/<trajectory>/<trajectory>.bag        raw ROS1 capture
  -> scripts/convert/bag_to_dataset_rosbags.py
qcar_onboard/<ip>/dataset/converted_<trajectory>/<frame>/    per-frame PNGs + Vicon pose
  -> scripts/build/cooperative/build_coop_train_val_dataset.py
datasets/CoopFront/{train,validate}/<scenario>/<cav_id>/<frame>.yaml + _camera0.png   the trained dataset
```

`cav_id` folders: `1` = qcar-52775 (`.198`), `2` = qcar-52776 (`.158`).
Only one real camera per agent is written (`camera0` = front) — HEAL's
`OPV2VBaseDataset.find_camera_files()` is hardcoded to expect exactly 4
paths, so `camera_attfuse_coop.yaml`'s config relies on the same
`patch_1cam_loader.py` monkeypatch `SmokeTestFront` used to need (see
`qcar/`).

## REAL vs ASSUMED, for `CoopFront` (verify against `build_coop_train_val_dataset.py` itself if in doubt)

| field | status |
|---|---|
| `lidar_pose` x,y / `vehicles[].location` x,y | **REAL** — Vicon, `valid=1.0` filtered, marker-to-physical-center offset applied (`OFFSET_FWD_M`/`OFFSET_LAT_M`, measured 2026-09-14) |
| `camera0.intrinsic` (both cars) | **REAL**, measured — `calibration_Matries/qcar5277{5,6}/latest_front/*_verified.npz`. qcar-52775's re-verified independently 2026-09-17 (see `calibration/calibration_Matries/REPORT.md`). |
| `camera0.cords` extrinsic, qcar-52775 (`cav_id "1"`) | **REAL**, measured — `R_CB_QCAR52775`/`T_CB_QCAR52775` in `build_coop_train_val_dataset.py`, from a 19-point image/Vicon correspondence fit (script/raw points themselves lost, only the resulting matrix survives) |
| `camera0.cords` extrinsic, qcar-52776 (`cav_id "2"`) | **ASSUMED** — `R_CB_NOMINAL` + Quanser manual mounting value, never independently measured for this car |
| `vehicles[].extent` | nominal QCar body, 0.425 x 0.192 x 0.190 m, not measured per-car |
| z, roll, pitch | zeroed in the written yaml (Vicon reports them but they aren't propagated) |
| static targets (vehicle ids 3, 4) | **REAL but fixed** — Vicon-measured once, held constant every frame; user-confirmed 2026-09-14 the two targets were physically parked there for the whole capture |
| wall occlusion (agent 1 x target 3, agent 2 x target 4) | **REAL physical fact**, user-confirmed 2026-09-14, hardcoded (`WALL_BLOCKED`) — no geometry check can derive a solid wall |
| pairing | nearest `wall_time_unix`, tolerance 40ms (`PAIR_TOLERANCE_SEC`) |
| world scale | x10 horizontal only (position/extent x,y); z, height, K untouched — real separation (0.65-3.0m) sits below the checkpoint's first depth bin (2m) unscaled |

## Reproducibility status of each dataset variant (checked 2026-09-18)

| dataset | reproducible from raw? | notes |
|---|---|---|
| `CoopFront/` | **Yes** | `qcar_onboard/<ip>/bags/*.bag` (raw) is present for all 9 trajectories used; re-run `bag_to_dataset_rosbags.py` + `build_coop_train_val_dataset.py` end to end any time. |
| `SmokeTestFront/`, `SmokeTestFront_dev/` (renamed 2026-09-18, formerly `OnlyFront`/`OnlyFront_dev`) | **No — historical, frozen** | Built from an earlier, smaller single-trajectory-pair capture (`Node3Via1To8`/`Node15Via6To8_02`, 2026-09-11) whose raw `.bag` was never present in this repo to begin with; cannot be regenerated. **Real, current utility, not just history:** it has the project's first-ever completed cooperative training run (`opencood/logs/HeterBaseline_opv2v_camera_attfuse_2026_09_12_18_27_33/`, checkpoint `net_epoch_bestval_at1.pth`, from `camera_attfuse_onlyfront.yaml`) — a working, independent baseline predating `CoopFront`. At 69MB total, it's also the fastest way to smoke-test a loader/patch change (`patch_1cam_loader.py`, `patch_real_extrinsic.py`) without waiting on the full `CoopFront` cycle. Renamed because "OnlyFront" stopped distinguishing anything once `CoopFront` also became single-camera; the new name reflects the actual reason to keep it. Structurally identical to `CoopFront` otherwise (same 2-agent architecture, same world-scale) — any NEW single-camera cooperative dataset should still go through `build_coop_train_val_dataset.py` on `qcar_onboard/`, not a revived builder here. `build_clean_dataset.py`/`to_opv2v_layout.py` (the scripts that built it) were removed 2026-09-18 since they only ever fed this now-frozen dataset. |
| `PretrainFront/` | **Yes, as of 2026-09-18** | `build_pretrain_dataset.py`/`build_pretrain_manifest.py` source from `/media/saavkd7/C4D26282D2627896/Concordia_QCar_Dataset/Scenarios/` (101 scenarios, confirmed present). That path is a real internal NTFS partition (`nvme1n1p3`, the machine's Windows dual-boot partition, 211.9GB) sharing free space on the internal drive — not a USB stick — so it was findable via `lsblk`/`blkid` once its `/media/saavkd7/C4D26282D2627896` mountpoint turned up empty. It is **not** in `/etc/fstab`, so it does not auto-mount on reboot; remount with `sudo mount -t ntfs-3g /dev/nvme1n1p3 /media/saavkd7/C4D26282D2627896` (create the mountpoint with `sudo mkdir -p` first) if this path goes missing again. |

## History

An earlier pipeline (`build_clean_dataset.py` + `to_opv2v_layout.py` ->
`clean_dataset/`, `opv2v_layout/`, `opv2v_ready/`) fed a `camera_attfuse_qcar.yaml`
config that was never actually trained (no matching run in `opencood/logs/`)
and also fed `SmokeTestFront/`'s build (see table above). Removed 2026-09-18 as
dead weight superseded by `build_coop_train_val_dataset.py` — see git history / 
the project's research notes (kept outside this repo; ask the maintainer) for the removed
version's own account of what was real vs assumed in that older layout.

Later the same day, the whole directory (`Inference/`) was renamed to
`pipeline/` and reorganized by type into `scripts/{convert,build,diagnose,visualize}/`,
`datasets/`, `shared/` (see "Layout of this directory" above) — the flat
`Inference/` name gave no indication of what the directory actually was.
Every cross-reference was updated: the 5 tracked yamls in
`opencood/hypes_yaml/opv2v/CameraOnly/qcar_real/` (added `/datasets/` to
`root_dir`/`validate_dir`/etc.), `opencood/qcar_patches/README.md`,
`HEAL-Concordia/CLAUDE.md`, and every script's own `HERE`/`PIPELINE_ROOT`
path constants and cross-script `sys.path.insert` calls (several scripts
import from siblings, e.g. `diagnose_*.py` importing `build_coop_train_val_dataset`,
which broke on the first pass and was caught by actually importing each
script after moving it, not just compiling it).

**2026-09-18, later still:** every hardcoded absolute path (`/mnt/mainvolume/...`,
including several stale `HEAL-Concordia/` ones from before that day's rename)
was replaced with `--flag`s defaulting to a path computed relative to the
repo — see `README.md`'s "Flag convention" for the full list and the pattern
to follow for new scripts. **A real incident happened during this pass**:
`build_coop_train_val_dataset.py --help` and the three `build_pretrain_*.py --help`
invocations, run to verify the new flags' `--help` text, actually executed
their full `main()` instead — `argparse.ArgumentParser(add_help=False)` means
`-h`/`--help` is silently ignored unless something else catches it, and
nothing did, so each script fell through to its unconditional
`shutil.rmtree()`-then-rebuild. `CoopFront/` was fully deleted and rebuilt
(verified afterward: identical output, 819 train / 237 validate, same
per-trajectory counts as before — no actual data loss, since the source
frames are deterministic). `PretrainFront/` was left partially built when the
process was killed mid-run; the partial directory was deleted rather than
kept in an ambiguous half-built state. `pretrain_manifest.json` was
untouched (killed before its own write). Fixed by adding an explicit
`if __name__ == "__main__" and any(a in ("-h","--help") for a in sys.argv[1:])`
guard before parsing in every script with this add_help=False pattern —
see `README.md`. Verified afterward with `timeout <n>` wrapped around every
`--help` invocation, not just a bare run.

**2026-09-23 — QCar framework code moved out of `opencood/` into `qcar/`.**
`opencood/qcar_patches/{train_qcar,eval_qcar,visualize_inference}.py` →
`qcar/{train,eval,visualize}.py`, the two monkeypatches →
`qcar/patches/`, `opencood/loss/point_pillar_depth_balanced_bce_loss.py` →
`qcar/losses/`, the 5 `qcar_real` yamls → `qcar/configs/` (all `git mv`,
HEAL repo root). `opencood/` is now byte-identical to upstream again.
Every `import opencood.qcar_patches.*` in `scripts/` and
`shared/evidence_through_wall/` (and the 4 notebooks) now reads
`qcar.patches.*`. Also fixed the stale `SmokeTestFront_dev` paths in
`camera_attfuse_onlyfront.yaml` / `camera_pyramid_onlyfront.yaml` (they
predated the move to `datasets/Smokes/`). Modules are now selectable by
config (`qcar/conf.json`, the yaml's `_qcar_*` keys) or CLI flag — see
`qcar/README.md`.
