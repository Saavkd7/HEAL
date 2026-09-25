# qcar/realtime — live cooperative inference from the QCars

Streams each car's ROS topics to this workstation, assembles frames, pairs
ego (agent 1, qcar-52775) with peer (agent 2, qcar-52776), writes each pair in
HEAL's input format and runs the fusion model, one pair at a time, newest first.

```
car roscore ──TCPROS──> ros1.py ──> sync.py (per car) ──> pairing ──> heal_input.py ──> HEAL model ──> jsonl
 /qcar/csi_front                    causal bounded wait    ≤ pair_tol    InferenceFront format
 /qcar/vicon                        (converter's logic)                  on /dev/shm
```

| file | role |
|---|---|
| `run.py` | entry point: subscribers, pairing, inference loop, stats, optional recording |
| `ros1.py` | pure-Python ROS1 subscriber (no ROS install; one process talks to both cars' masters) |
| `sync.py` | live port of `bag_to_dataset_rosbags.py`'s frame sync; **imports** its functions |
| `heal_input.py` | writes the pair via `build_inference.py`'s **imported** `build_agent_yaml`/calibration; builds HEAL dataset + model like `qcar/eval.py` |
| `fake_car.py` | replays recorded `.bag`s as live roscores, for testing without the cars |
| `conf.json` | every default (flags override for one run) |

No existing script is modified. Sync settings (`fps`, `wait_sec`, tolerances,
`vicon_pose_key`) come from the converter's `conf.json`; `pair_tolerance_sec`,
car identity and calibration come from `build_inference.py`. So live frames use
the same settings that built the training data.

## Run

```bash
python -m qcar.realtime.run                 # real cars (masters in conf.json "cars")
python -m qcar.realtime.run --no_model      # link check: receive + sync + pair + format only
python -m qcar.realtime.run --record_dir data_live   # also save pairs in converter layout

# without cars
python -m qcar.realtime.fake_car &
python -m qcar.realtime.run --master 1=http://127.0.0.1:11411 --master 2=http://127.0.0.1:11412
```

Output: one JSON line per fused pair in `opencood/logs/realtime/<session>.jsonl`:
detections (`corners_ego_model` in HEAL's x10 ego frame, `center_world_m` in the
Vicon frame), scores, both poses, the pairing offset, and latency broken down by stage.

## Requirements and caveats

- **CUDA required.** HEAL's camera encoder calls `.cuda()` when it's built.
- **Car clocks must be synchronized** (NTP/chrony). Pairing compares the two
  cars' ROS stamps, just as the offline dataset does. `[stats]` prints each car's
  median `arrival - stamp`; a gap between the two cars means clock skew.
- **Bandwidth.** Raw `sensor_msgs/Image` front camera ≈ 0.9 MB/msg at 20 Hz ≈
  18 MB/s per car. The default subscribes to front + vicon only, and each extra camera adds as much again.
- If a car advertises an unresolvable hostname (`ROS_HOSTNAME` unset), the
  subscriber falls back to that car's master IP.
- `/dev/shm/qcar_realtime` is wiped at start.

## Verified (2026-09-25, fake_car replay of trajectory `_00`)

- HEAL batches built from the live formatter are **identical** (30/30, max diff 0)
  to those built from `build_inference.py`'s offline output for the same frames.
- Live front images are byte-identical to the converted dataset. Vicon poses are
  identical or one adjacent sample away (≤7 mm), which comes from causal selection
  by arrival time.
- Model forward on the live stream not yet run: the GPU was occupied during testing.
