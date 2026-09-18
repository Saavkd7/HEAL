# `opencood/` — what's upstream HEAL/OpenCOOD, what's this project's own addition

Everything here is **upstream HEAL/OpenCOOD** (`main`, tracks
`upstream/yifanlu0227/HEAL` cleanly) **except** the QCar real-testbed
integration, which is confined to exactly these paths — confirmed with
`git diff main --stat -- opencood/` (12 files, all pure additions, zero
upstream files modified):

```
opencood/
  qcar_patches/                                    ← QCar-only, entirely new (6 files)
  loss/point_pillar_depth_balanced_bce_loss.py      ← QCar-only, entirely new
  hypes_yaml/opv2v/CameraOnly/qcar_real/*.yaml      ← QCar-only, 5 configs, entirely new
  logs/                                             ← gitignored; QCar's own trained runs
                                                       live here (see logs/README.md), plus
                                                       upstream's own generic setup file
  (everything else: data_utils/, models/, tools/,
   utils/, hypes_yaml/<other datasets>/, ...)       ← upstream HEAL, untouched by this branch
```

**Start here for the QCar work:** `qcar_patches/README.md` — the full map of
what each QCar-specific file does and how they're invoked. This file is only
the top-level orientation.

**Checkpoints:** `../checkpoints/` at the repo root (not inside `opencood/`)
— official HEAL baseline downloads, split into `qcar/` (the one thing QCar
configs actually reference) and `heal_reference/` (everything else, general
framework material unrelated to QCar). See its own `README.md`.

**Why nothing upstream is touched:** `main` is kept as a clean, rebasable
diff against upstream on purpose (see `../HEAL-Concordia/CLAUDE.md`) — the
QCar integration is designed to be pure addition (new files, new yaml
configs, monkeypatches applied only when a QCar script imports them), never
an edit to vendored HEAL source. If you ever find yourself about to edit a
file outside the list above to make QCar work, that's a sign to write a
patch/wrapper instead, matching the existing pattern in `qcar_patches/`.
