# `qcar_onboard/` — raw Vicon+camera captures for the QCar/HEAL cooperative dataset

One folder per car, **named by its IP**. As of 2026-09-18 this directory only
holds the two cars used by the offline Vicon-capture → HEAL/OPV2V pipeline
(`CoopFront`): `192.168.1.198` (qcar-52775) and `192.168.1.158` (qcar-52776,
the cooperating peer). The lab has two other QCars (`192.168.0.104`/`.144`)
used by a separate, unrelated on-device live-fusion project (TensorRT,
no Vicon, no `.bag`) — their mirror was removed from here entirely on
2026-09-17 since it had no relation to this pipeline.

## Layout, per car

```
192.168.1.<ip>/
  bags/                             raw ROS1 .bag files only
    MovingFourCameraVicon_<node>_00/
      MovingFourCameraVicon_<node>_00.bag
    ... 00-08 (9 trajectories)
  dataset/                          converted per-frame output only
    converted_MovingFourCameraVicon_<node>_00/
      000000/front.png back.png left.png right.png
             ego_vicon_pose.json timestamp.json
      ...
      metadata.json
    ... 00-08 (9 trajectories)
```

**`bags/` and `dataset/` are deliberately split** (reorganized 2026-09-18 for
clarity — an earlier layout mixed raw `.bag`s and converted frames inside one
`bags/` folder, and kept a second, differently-produced `dataset/` folder
alongside it, which was confusing enough to cause real mistakes reading it).
The rule now: **`bags/` is raw, unprocessed ROS capture. `dataset/` is what
`build_coop_train_val_dataset.py` actually reads to build `CoopFront`.** Nothing else
in this directory is used by training.

### `dataset/converted_*` is a re-conversion done on this workstation

Each `converted_<trajectory>/` folder was produced by running
`bag_to_dataset_rosbags.py` against the matching `bags/<trajectory>/*.bag`,
**on this machine** — confirmed by each `converted_*/metadata.json`'s
`source_bag` field, which points at the local `bags/` path, not the
Jetson's. This is the only converted output kept in this repo; see "What got
deleted" below for what used to sit alongside it.

### `build_coop_train_val_dataset.py`'s paths were stale — fixed 2026-09-18

`data/qcar_dataset/Inference/build_coop_train_val_dataset.py`'s `BAGS_198`/`BAGS_158`
constants used to point at
`/mnt/mainvolume/Backup/Projects/Concordia/Cooperative_Perception/qcar_onboard/...`
— a path that no longer exists on this machine (the data had already been
copied into this repo's `qcar_onboard/` before this was noticed). The script
would have failed immediately if anyone tried to rebuild `CoopFront` from
scratch. Fixed to point at `data/qcar_onboard/192.168.1.<ip>/dataset`
(matches the layout above — `build_coop_train_val_dataset.py` only ever reads
`<root>/converted_<prefix><idx>/`, i.e. things directly under `dataset/`).

## What got deleted (2026-09-18) — read this before looking for trajectory `_09`

Before the split above, each car also had a **second, independent**
per-frame conversion, produced **on the QCar's own Jetson**
(`ego_vicon_pose.json`'s `metadata.json.source_bag` pointed at
`/home/nvidia/Collection/qcar_bags/...`). It existed for **10** trajectories
(`_00`-`_09`) per car, versus the workstation conversion's 9
(`build_coop_train_val_dataset.py` hardcodes `N_PAIRS = 9`, `_00`-`_08` only).

This onboard duplicate was deleted at the user's request on 2026-09-18 to
stop the two similarly-shaped folders from being confused with each other
(previously named `dataset/` on `.198` and, briefly, `10-Trajectories/` on
`.158`). **Consequence: trajectory `_09`'s data no longer exists anywhere in
this repo, for either car** — it was never used by `CoopFront` and its raw
`.bag` was never kept either way, but if you need to re-investigate the open
question in `PENDIENTE` about why the "10th trajectory" was never folded in,
you'd need to re-pull it fresh from the Jetson (`/home/nvidia/Collection/qcar_bags/`
on each car) — assuming it's still there.

The deleted folder was also the only evidence that the onboard conversion and
the workstation conversion produce **different frame counts** for the same
`.bag` (non-deterministic causal-bounded-wait sync) — that finding is now
written up in `11 EVIDENCIA` (the project's research notes (kept outside this repo; ask the maintainer)) so it isn't lost, but the raw
comparison data itself is gone.

## Tools (`_tools/`)

### `pull_qcar_bags.py` — pull a car's raw `.bag`s over SSH, any QCar
Written 2026-09-17 to replace a transfer script that was lost
(`sync_198.sh`/`pull_all_bags.sh`, never saved to any repo). Pulls a car's
raw Vicon-capture `.bag` directory over SSH+tar into `<car>/bags/` — read-only
on the vehicle, nothing installed/moved/deleted on the car side. Works
against any QCar IP, not just `.198`/`.158`; pass `--remote-dir` if a car
keeps its bags somewhere other than `_tools/conf.json`'s `remote_dir`
(`/home/nvidia/Collection/qcar_bags`, confirmed for qcar-52775/52776). Every
default (host `192.168.1.198`, user, remote dir, local destination
`local_root` = `Training/ExperimentNo1/`) lives in `_tools/conf.json`; run with
no args or override per run. Uses SSH keys (`password: null`); pass
`--password` if a car isn't key-authenticated yet -- never store it in conf.json. Needs `paramiko` (not in this project's `heal38` env; available in
`qlab`, or install it wherever you run this). **Never actually run against a
live car yet** — the QCars weren't reachable from this machine's network
during development, so treat it as unverified against real hardware even
though its syntax is checked and its SSH+tar mechanism mirrors the one
already proven for the other two QCars.
```bash
python3 pull_qcar_bags.py 192.168.1.198
python3 pull_qcar_bags.py 192.168.1.158 --password nvidia        # if keys aren't set up on this machine
python3 pull_qcar_bags.py 192.168.1.198 --manifest-only          # re-probe only, pull nothing
```

### `cross_check_vicon_labeling.py` — check the ego/target Vicon mislabeling bug
Written 2026-09-18 to answer an open question from `PENDIENTE`: does the
documented (but unfixed) ego/target Vicon mislabeling bug
(`bag_to_dataset_rosbags.py`'s own docstring) affect any of the 9 trajectories
`CoopFront` actually uses? Needs `rosbags`, present in the `heal38` env:
```bash
~/.pyenv/versions/heal38/bin/python cross_check_vicon_labeling.py
```
**Finding:** no. `.198` resolves `vicon_pose_key = "ego"` and `.158` resolves
`"target"` in all 9 trajectories, with zero deviation — the roles never flip
between sessions, so none of the 9 trajectories' `ego_vicon_pose.json`
secretly holds the wrong car's position. Full reasoning (including a dead
end: an earlier version of this script wrongly assumed each car
self-labels "ego" vs. "peer", which produced a false "MISLABELED" verdict on
every trajectory — the real architecture is one shared Vicon broadcast with
fixed global roles for the whole session, not a per-car self/peer split) is
in the script's own docstring. Run with `--diagnostic` to
see the raw ego/target distance table that revealed the shared-broadcast
architecture — that mode is for inspecting the broadcast claim itself, not a
bug detector.

## Identifying a car

A car's stable identity is its hostname/serial/MAC, not its IP (DHCP). `pull_qcar_bags.py` writes a
`BAGS_MANIFEST.md` per car (hostname, serial, MAC, sha256 of what was pulled);
the ones for the `Inference/` captures exist locally but are not in git, since
they carry the lab cars' device identifiers. If an IP ever changes,
rename the folder and record the mapping here.

## Full QCar/HEAL pipeline context

For the end-to-end story of how these `.bag`s become a trainable dataset,
see `../../SCRIPTS.md` (every script, in pipeline order) and
`../qcar_dataset/pipeline/README.md`; the narrative write-up is `10 PIPELINE`
(with `11 EVIDENCIA`, `12 EVIDENCIA`) in the project's research notes (kept outside this repo; ask the maintainer).
