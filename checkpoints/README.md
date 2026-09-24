# Checkpoints — official HEAL baselines, NOT this project's own results

**Everything in this directory is a pre-existing download from HEAL's official
HuggingFace hub** (`https://huggingface.co/yifanlu/HEAL`), fetched 2026-09-09.
See `MANIFEST.md` for exact source URLs, byte sizes, and SHA-256 checksums.
**None of this was trained by this project.**

This project's own trained checkpoints (the QCar work) live entirely
elsewhere: `opencood/logs/<run_name>/net_epoch_bestval_at*.pth` (see that
directory's own `README.md`) — currently two real completed runs:
- `HeterBaseline_opv2v_camera_attfuse_2026_09_12_18_27_33` — trained on
  `SmokeTestFront_dev` (the project's first working cooperative checkpoint).
- `HeterBaseline_opv2v_camera_attfuse_qcar_coop_2026_09_14_12_37_23` — trained
  on `CoopFront` (the current main dataset).

## Layout, reorganized 2026-09-18 by actual QCar relevance

```
checkpoints/
  qcar/                 the ONE checkpoint QCar configs actually reference
    HeterBaseline_opv2v_camera_attfuse_2023_08_08_16_50_01/
  heal_reference/        everything else — general HEAL framework material,
                          zero relation to the QCar work, kept intentionally
    opv2v_camera/
      HeterBaseline_opv2v_camera_disco_2023_08_08_16_50_01/
      HeterBaseline_opv2v_camera_fcooper_2023_08_06_11_48_21/
      HeterBaseline_opv2v_camera_v2xvit_2023_08_07_04_46_10/
    dair_camera/
      HeterBaseline_DAIR_camera_attfuse_2023_09_09_11_24_40/
      HeterBaseline_DAIR_camera_cobevt_2023_09_09_11_25_45.zip   (unextracted)
      HeterBaseline_DAIR_camera_disco_2023_09_09_11_27_56.zip    (unextracted)
      HeterBaseline_DAIR_camera_fcooper_2023_09_09_11_28_21.zip  (unextracted)
      HeterBaseline_DAIR_camera_v2xvit_2023_09_09_11_27_38.zip   (unextracted)
```

### `qcar/` — the only one loaded by any config in this repo

`HeterBaseline_opv2v_camera_attfuse_2023_08_08_16_50_01` (simplest attention
fusion baseline, camera-only, trained on simulated OPV2V) — its
`encoder_m2.*` weights are transplanted as the encoder-initialization source
for `camera_attfuse_pretrain.yaml` / `camera_attfuse_onlyfront.yaml` /
`camera_pyramid_onlyfront.yaml` / `camera_attfuse_qcar.yaml`'s
`_qcar_pretrained_checkpoint` / `_qcar_encoder_initialization_checkpoint`
fields (paths updated 2026-09-18 to `checkpoints/qcar/...` — see
`qcar/configs/*.yaml`).

### `heal_reference/` — general HEAL framework material, not QCar's problem

Kept deliberately (not deleted) after confirming with the user that this
repo also serves as a general HEAL/OpenCOOD framework fork, not only the
QCar integration. `opv2v_camera/{disco,fcooper,v2xvit}` and all of
`dair_camera/` exist to reproduce HEAL's own published benchmark numbers on
simulated OPV2V and real DAIR-V2X-C (via `opencood/hypes_yaml/dairv2x/CameraOnly/*.yaml`,
upstream configs, unmodified by this branch) — **nothing here is referenced
by any QCar-specific file.** `dair_camera`'s 4 non-attfuse methods are left
as their original `.zip` (never extracted) since they're the only copy.

## History

**2026-09-18, cleanup pass 1:** deleted 5 `.zip` files (2.6GB) that were pure
duplicates of folders already extracted right next to them (`opv2v_camera`'s
4, `dair_camera`'s `attfuse`) — `MANIFEST.md`'s checksums still describe the
originals if ever needed again (re-download from the HuggingFace URL above).

**2026-09-18, cleanup pass 2:** reorganized into `qcar/` vs `heal_reference/`
(layout above) once it became clear "everything not referenced by a QCar
config" wasn't actually clutter this project created — it's separate,
legitimate general-framework reference material that happened to live
alongside the QCar-relevant checkpoint with no label distinguishing them.
Updated the 4 yaml path references accordingly; verified the referenced
`.pth` file resolves at its new path.

Each extracted run folder's own `config.yaml` (git-tracked, see the root
`.gitignore`'s `!checkpoints/**/config.yaml` exception) records exactly what
was trained — read that before assuming what a checkpoint's architecture is.
