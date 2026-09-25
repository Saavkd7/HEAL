"""Real-time cooperative inference: QCar ROS topics -> HEAL fusion, live.

    python -m qcar.realtime.run                       # every default from qcar/realtime/conf.json
    python -m qcar.realtime.run --no_model            # receive + format only (check the link)
    python -m qcar.realtime.run --no_view             # no windows (headless / over SSH)
    python -m qcar.realtime.run --master 1=http://127.0.0.1:11411 --master 2=http://127.0.0.1:11412
                                                      # against qcar/realtime/fake_car.py

Pipeline, per car (both run concurrently):
  1. ros1.TopicSubscriber streams /qcar/csi_front (+ other cameras if asked)
     and /qcar/vicon straight from that car's roscore (no ROS install here).
  2. sync.CarSync assembles frames with bag_to_dataset_rosbags.py's own
     causal bounded-wait logic -> the same content as a converted frame
     folder (front.png + ego_vicon_pose.json + timestamp.json).
Then, across cars:
  3. the newest ego frame (agent 1) is paired with the peer frame (agent 2)
     nearest in source time, within build_inference.py's pair_tolerance_sec.
     Older unprocessed frames are skipped -- real time means latest wins.
  4. heal_input.LiveHealInput writes the pair in build_inference.py's exact
     InferenceFront format (on tmpfs), and heal_input.LiveModel pushes it
     through HEAL's own dataset + the chosen model + decode/NMS.
  5. one JSON line per fused pair goes to output_jsonl: detections (ego frame,
     model space, and world metres), scores, timing, pairing offsets.

Cross-car pairing compares the two cars' own ROS stamps, exactly like the
offline dataset; the cars' clocks must be synchronized (NTP/chrony) for it
to mean anything. The periodic stats print each car's (arrival - stamp) so a
clock offset is visible instead of silently breaking the pairing.
"""
import argparse
import collections
import datetime
import json
import os
import queue
import statistics
import sys
import tempfile
import threading
import time

import cv2
import numpy as np

from qcar import config
from qcar.realtime import heal_input, ros1
from qcar.realtime.sync import CarSync

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONF = os.path.join(HERE, "conf.json")


def realtime_conf_path():
    argv = sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg == "--realtime_conf" and i + 1 < len(argv):
            return os.path.abspath(argv[i + 1])
        if arg.startswith("--realtime_conf="):
            return os.path.abspath(arg.split("=", 1)[1])
    return DEFAULT_CONF


def parse_args():
    rt = config.load_conf(realtime_conf_path())
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--realtime_conf", default=DEFAULT_CONF,
                    help="this script's conf.json")
    ap.add_argument("--model_dir", default=None,
                    help="trained run dir with resolved_hypes.json (default: conf model_dir)")
    ap.add_argument("--master", action="append", default=[], metavar="AGENT=URI",
                    help="override one car's ROS master, e.g. 1=http://192.168.1.198:11311")
    ap.add_argument("--cameras", default=None,
                    help="comma list of cameras to subscribe to (default: conf cameras)")
    ap.add_argument("--live_dir", default=None, help="tmpfs dir HEAL reads (default: conf)")
    ap.add_argument("--record_dir", default=None,
                    help="also save every fused pair in converter layout (default: conf)")
    ap.add_argument("--output_jsonl", default=None, help="default: conf output_jsonl")
    ap.add_argument("--amp", dest="amp", action="store_true")
    ap.add_argument("--no_amp", dest="amp", action="store_false")
    ap.set_defaults(amp=rt["amp"])
    ap.add_argument("--no_view", dest="view", action="store_false",
                    help="do not open the live viewer (default: conf view)")
    ap.set_defaults(view=rt["view"])
    ap.add_argument("--no_model", action="store_true",
                    help="receive, pair and format only; do not load/run HEAL")
    ap.add_argument("--max_pairs", type=int, default=None,
                    help="stop after this many fused pairs (testing)")
    config.add_module_args(ap)  # --conf (qcar/conf.json), --plugins, --set, ...
    opt = ap.parse_args()

    opt.model_dir = config.cli_path(opt.model_dir) if opt.model_dir else rt.path_of("model_dir")
    opt.live_dir = config.cli_path(opt.live_dir) if opt.live_dir else rt.path_of("live_dir")
    if sys.platform.startswith("win") and opt.live_dir.replace("\\", "/").startswith("/dev/shm"):
        # no tmpfs on Windows: use the user's temp dir instead of C:\dev\shm
        opt.live_dir = os.path.join(tempfile.gettempdir(), "qcar_realtime")
    opt.record_dir = (config.cli_path(opt.record_dir) if opt.record_dir
                      else rt.path_of("record_dir"))
    opt.output_jsonl = (config.cli_path(opt.output_jsonl) if opt.output_jsonl
                        else rt.path_of("output_jsonl"))
    opt.cameras = config.split_list(opt.cameras) if opt.cameras else list(rt["cameras"])
    opt.masters = dict((k, v["master_uri"]) for k, v in rt["cars"].items())
    for item in opt.master:
        aid, _, uri = item.partition("=")
        opt.masters[aid.strip()] = uri.strip()
    return opt, rt


def box_record(corners, score, ego_pose):
    """One detection: ego-frame corners in model space (x10 world, what HEAL
    outputs) plus the centre in world metres (Vicon frame)."""
    center = corners.mean(axis=0)
    c, s = np.cos(ego_pose["yaw"]), np.sin(ego_pose["yaw"])
    wx = (c * center[0] - s * center[1]) / 10.0 + ego_pose["x"]
    wy = (s * center[0] + c * center[1]) / 10.0 + ego_pose["y"]
    return {"score": float(score), "corners_ego_model": np.round(corners, 4).tolist(),
            "center_world_m": [round(float(wx), 4), round(float(wy), 4)]}


class Recorder(threading.Thread):
    """Saves fused pairs in bag_to_dataset_rosbags.py's per-frame layout,
    off the inference thread (PNG encoding is slow)."""

    def __init__(self, root, session, bi, conv, syncs):
        threading.Thread.__init__(self, daemon=True)
        self.q = queue.Queue(maxsize=64)
        self.dirs = dict((aid, os.path.join(root, bi.CAR_REGISTRY[aid]["ip"], "dataset",
                                            "converted_live_%s" % session))
                         for aid in bi.CAR_REGISTRY)
        self.conv, self.syncs = conv, syncs
        self.dropped = 0

    def put(self, pair_id, frames):
        try:
            self.q.put_nowait((pair_id, frames))
        except queue.Full:
            self.dropped += 1

    def run(self):
        while True:
            pair_id, frames = self.q.get()
            for aid, fr in frames.items():
                d = os.path.join(self.dirs[aid], "%06d" % pair_id)
                os.makedirs(d, exist_ok=True)
                for name, msg in fr["images"].items():
                    img = fr["front_bgr"] if name == "front" else self.syncs[aid].decode_image(msg)
                    cv2.imwrite(os.path.join(d, name + ".png"), img,
                                [cv2.IMWRITE_PNG_COMPRESSION, 0])
                self.conv.atomic_json(os.path.join(d, "ego_vicon_pose.json"), fr["ego_pose"])
                ts = dict(fr["timestamp"], frame_id="%06d" % pair_id)
                self.conv.atomic_json(os.path.join(d, "timestamp.json"), ts)
            self.q.task_done()


def main():
    opt, rt = parse_args()
    qconf = config.load_conf(opt.conf)
    os.chdir(config.REPO_ROOT)
    session = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    bi = heal_input.import_script(rt.path_of("build_inference_py"), "qcar_rt_build_inference")
    conv = heal_input.import_script(rt.path_of("converter_py"), "qcar_rt_bag_to_dataset")
    sync_conf = json.load(open(rt.path_of("convert_conf")))
    agents = sorted(bi.CAR_REGISTRY)
    ego_id, peer_id = bi.EGO_ID, bi.PEER_ID
    missing = [a for a in agents if a not in opt.masters]
    if missing:
        raise SystemExit("no ROS master configured for agent(s) %s" % missing)
    pair_tol = float(bi.PAIR_TOLERANCE_SEC)
    pair_wait = float(rt["pair_wait_sec"])

    print("[realtime] session %s | ego=%s peer=%s | cameras=%s | fps=%s wait=%.3fs "
          "pair_tol=%.3fs" % (session, ego_id, peer_id, opt.cameras, sync_conf["fps"],
                               sync_conf["wait_sec"], pair_tol))

    live = heal_input.LiveHealInput(bi, opt.live_dir)
    syncs = dict((a, CarSync(conv, opt.cameras, sync_conf, rt["max_network_lag_sec"]))
                 for a in agents)
    subs = []
    for aid in agents:
        for name in syncs[aid].cameras + ["vicon"]:
            cb = (lambda s, n: lambda raw, mt, arr: s.push(n, raw, mt, arr))(syncs[aid], name)
            sub = ros1.TopicSubscriber(opt.masters[aid], rt["topics"][name], cb,
                                       float(rt["reconnect_sec"]))
            sub.agent, sub.name = aid, name
            sub.start()
            subs.append(sub)
        print("[realtime] agent %s (%s) <- %s" % (aid, bi.CAR_REGISTRY[aid]["car_label"],
                                                  opt.masters[aid]))

    recorder = None
    if opt.record_dir:
        recorder = Recorder(opt.record_dir, session, bi, conv, syncs)
        recorder.start()
        print("[realtime] recording to %s" % opt.record_dir)
    out = None
    if not opt.no_model:
        os.makedirs(os.path.dirname(opt.output_jsonl.format(session=session)), exist_ok=True)
        out = open(opt.output_jsonl.format(session=session), "w")

    viewer = None
    if opt.view:
        if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
            print("[realtime] no DISPLAY -- viewer disabled (use --no_view to silence this)")
        else:
            from qcar.realtime.viewer import LiveViewer
            walls = json.load(open(bi._path_from_conf("walls_json")))["walls"]
            viewer = LiveViewer(agents, ego_id,
                                dict((a, bi.CAR_REGISTRY[a]["car_label"]) for a in agents),
                                dict((a, bi.CAM_PARAMS[a]["K"]) for a in agents),
                                walls, rt["view_bev_range_m"], rt["view_max_hz"])
            print("[realtime] viewer open: 2 POV windows + bird's-eye view (q to stop)")

    model = None
    queues = dict((a, collections.deque(maxlen=64)) for a in agents)
    stats = collections.Counter()
    latencies = collections.deque(maxlen=200)
    skew = dict((a, collections.deque(maxlen=200)) for a in agents)
    last_stats = time.time()
    last_bytes = dict((id(s), 0) for s in subs)
    n_pairs = 0

    try:
        while opt.max_pairs is None or n_pairs < opt.max_pairs:
            now = time.time()
            for aid in agents:
                for fr in syncs[aid].poll(now):
                    queues[aid].append(fr)
                    skew[aid].append(fr["arrival"] - fr["stamp"])

            pair = None
            eq, pq = queues[ego_id], queues[peer_id]
            for i in range(len(eq) - 1, -1, -1):
                e = eq[i]
                cands = [p for p in pq if abs(p["stamp"] - e["stamp"]) <= pair_tol]
                if cands:
                    pair = (e, min(cands, key=lambda p: abs(p["stamp"] - e["stamp"])))
                    stats["skipped_stale_ego"] += i
                    for _ in range(i + 1):
                        eq.popleft()
                    while pq and pq[0]["stamp"] <= pair[1]["stamp"]:
                        pq.popleft()
                    break
            while eq and pair is None and now - eq[0]["arrival"] > pair_wait:
                eq.popleft()
                stats["dropped_no_peer"] += 1
            while pq and now - pq[0]["arrival"] > pair_wait + 1.0:
                pq.popleft()

            if viewer is not None:
                if viewer.quit:
                    break
                if n_pairs == 0 and now - viewer.last_draw >= 1.0:
                    viewer.waiting("Esperando datos... frames: %s | pares: 0" % ", ".join(
                        "CAV %s=%d" % (a, syncs[a].frames) for a in agents))
                    viewer.last_draw = now
                viewer.pump()
            if pair is None:
                time.sleep(0.002)
            else:
                frames = {}
                for aid, fr in zip((ego_id, peer_id), pair):
                    fr["front_bgr"] = syncs[aid].decode_image(fr["images"]["front"])
                    frames[aid] = fr
                t_write = time.perf_counter()
                live.write_pair(frames)
                format_s = time.perf_counter() - t_write
                if recorder is not None:
                    recorder.put(n_pairs, frames)
                if not opt.no_model:
                    if model is None:
                        print("[realtime] first pair formatted -- loading model %s" % opt.model_dir)
                        if viewer is not None:
                            viewer.waiting("Primer par recibido - cargando modelo HEAL...")
                        model = heal_input.LiveModel(opt.model_dir, opt.live_dir, qconf, opt,
                                                     opt.amp)
                        print("[realtime] model on %s, amp=%s" % (model.device, model.amp))
                    boxes, scores, timing = model.infer()
                    done = time.time()
                    e, p = pair
                    rec = {
                        "pair": n_pairs,
                        "ego_stamp": e["stamp"], "peer_stamp": p["stamp"],
                        "pair_offset_sec": p["stamp"] - e["stamp"],
                        "latency_from_last_arrival_sec": done - max(e["arrival"], p["arrival"]),
                        "latency_from_ego_capture_sec": done - e["stamp"],
                        "format_s": format_s, **timing,
                        "ego_pose": e["ego_pose"], "peer_pose": p["ego_pose"],
                        "detections": [box_record(b, s, e["ego_pose"])
                                       for b, s in zip(boxes, scores)],
                    }
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                    latencies.append(rec["latency_from_last_arrival_sec"])
                    stats["detections"] += len(scores)
                if viewer is not None and viewer.due():
                    if opt.no_model:
                        viewer.update(frames, np.zeros((0, 8, 3)), np.zeros(0),
                                      {"pair": n_pairs, "status": "sin modelo (--no_model)"})
                    else:
                        viewer.update(frames, boxes, scores, {
                            "pair": n_pairs, "n_agents_fused": timing.get("n_agents_fused", "-"),
                            "status": "latencia %.0f ms | modelo %.0f ms" % (
                                rec["latency_from_last_arrival_sec"] * 1000,
                                timing.get("model_s", 0) * 1000)})
                n_pairs += 1
                stats["pairs"] += 1

            if now - last_stats >= float(rt["stats_every_sec"]):
                dt = now - last_stats
                parts = []
                for s in subs:
                    rate = (s.bytes - last_bytes[id(s)]) / dt / 1e6
                    last_bytes[id(s)] = s.bytes
                    parts.append("%s/%s %d msgs %.1fMB/s" % (s.agent, s.name, s.messages, rate))
                print("[stats] " + " | ".join(parts))
                for aid in agents:
                    med = statistics.median(skew[aid]) if skew[aid] else float("nan")
                    print("[stats] agent %s: %d frames, arrival-stamp median %.3fs, rejected %s"
                          % (aid, syncs[aid].frames, med, dict(syncs[aid].rejections)))
                lat = statistics.median(latencies) if latencies else float("nan")
                print("[stats] pairs %d, dropped_no_peer %d, skipped_stale %d, "
                      "median latency %.3fs%s" % (stats["pairs"], stats["dropped_no_peer"],
                                                  stats["skipped_stale_ego"], lat,
                                                  (", recorder dropped %d" % recorder.dropped)
                                                  if recorder else ""))
                last_stats = now
    except KeyboardInterrupt:
        pass
    finally:
        for s in subs:
            s.stop()
        if recorder is not None:
            recorder.q.join()  # finish writing queued pairs before exiting
        if out is not None:
            out.close()
            print("[realtime] detections: %s" % out.name)
        print("[realtime] %d pairs fused, %s" % (n_pairs, dict(stats)))


if __name__ == "__main__":
    main()
