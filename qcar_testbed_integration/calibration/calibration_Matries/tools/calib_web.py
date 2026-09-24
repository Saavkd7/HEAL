"""Live fisheye-calibration assistant for the QCar, driven from a phone browser.

Why this exists: capturing blind in 25 s takes and grading afterwards wasted
several passes. This closes the loop -- it shows, live, which part of the lens
is still unconstrained and tells the operator where to move the board.

    python3 calib_web.py --car 192.168.1.198 --camera front
    then open http://<workstation-ip>:8000 on a phone

Coverage is tracked in INCIDENCE ANGLE, not in image pixels: what matters is
whether the distortion polynomial is constrained out to the lens FOV. The 8x8
image grid was a poor proxy -- the outer ring is a thin band in pixels but a
large span in degrees.
"""
import argparse
import json
import os
import socket
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

try:
    from http.server import ThreadingHTTPServer
except ImportError:  # Python 3.6 on the Jetson TX2
    from socketserver import ThreadingMixIn

    class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))  # tools/ -> calibration_Matries/ -> calibration/ -> qcar_testbed_integration/
DEFAULT_CONF = os.path.join(HERE, "conf.json")


class _Conf(dict):
    """conf.json contents. It is THE source of every default -- nothing is
    hardcoded in this script; a CLI flag only overrides a key for one run.
    A missing key is a clear error, never a silent fallback."""

    def __init__(self, path, data):
        dict.__init__(self, data)
        self.path = path

    def __missing__(self, key):
        raise SystemExit("%s has no %r key -- add it there" % (self.path, key))

    def path_of(self, key):
        """Path-valued key: relative paths resolve against the
        qcar_testbed_integration/ root (REPO_ROOT); null stays None."""
        v = self[key]
        return v if not v or os.path.isabs(v) else os.path.join(REPO_ROOT, v)


def _load_conf(path):
    if not os.path.exists(path):
        raise SystemExit("conf.json not found: %s" % path)
    with open(path) as f:
        return _Conf(path, json.load(f))



_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--conf", default=DEFAULT_CONF)
_conf_path = _pre.parse_known_args()[0].conf
_conf = _load_conf(_conf_path)

ap = argparse.ArgumentParser()
ap.add_argument("--conf", default=DEFAULT_CONF,
                 help="Path to the conf.json to read defaults from (default: %s)" % DEFAULT_CONF)
ap.add_argument("--car", default=_conf["calib_car"])
ap.add_argument("--camera", default=_conf["calib_camera"])
ap.add_argument("--car-port", type=int, default=_conf["calib_car_port"])
ap.add_argument("--http-port", type=int, default=_conf["calib_http_port"])
ap.add_argument("--out", default=_conf.path_of("calib_out"), help="where to save accepted views; may use {car}/{camera} (default: conf.json's calib_out)")
ap.add_argument("--cols", type=int, default=_conf["calib_cols"])
ap.add_argument("--rows", type=int, default=_conf["calib_rows"])
ap.add_argument("--seed", default=_conf.path_of("calib_seed"), help="folder of earlier good views to preload")
ap.add_argument("--seed-state", default=_conf.path_of("calib_seed_state"),
                help="JSON state from an earlier assistant; loads instantly")
ap.add_argument("--capture-interval", type=float, default=_conf["calib_capture_interval"],
                help="minimum seconds between accepted views")
ap.add_argument("--novelty-px", type=float, default=_conf["calib_novelty_px"],
                help="minimum mean corner motion before accepting another view")
ap.add_argument("--square-size", type=float, default=_conf["calib_square_size"],
                help="checker square size in metres")
ap.add_argument("--initial-intrinsics", default=_conf.path_of("calib_initial_intrinsics"),
                help="NPZ providing CAMERA_K/D until the live fit is ready")
ap.add_argument("--extrinsics", default=_conf.path_of("calib_extrinsics"),
                help="NPZ providing fixed CAMERA_R_camera_to_vehicle/t_vehicle_m")
A = ap.parse_args()
if A.capture_interval < 0 or A.novelty_px < 0 or A.square_size <= 0:
    ap.error("capture interval/novelty must be non-negative and square size positive")

OUT = A.out.format(car=A.car.replace('.', '_'), camera=A.camera)
os.makedirs(OUT, exist_ok=True)

W, H = 640, 480
F_GUESS = 332.0                  # only to convert pixels->degrees for guidance
CX, CY = W / 2.0, H / 2.0
N_SECT, N_RING = 12, 4           # 12 sectors x 4 rings of incidence angle
RING_EDGES = [0, 20, 35, 50, 70] # degrees
MIN_VIEWS_PER_ZONE = 3
ANGLE_GOALS_DEG = np.array([65.0, 65.0, 60.0, 60.0])
ANGLE_TOLERANCE_DEG = 0.2
CORNER_NAMES = ("UPPER LEFT", "UPPER RIGHT", "LOWER LEFT", "LOWER RIGHT")
CORNER_TARGETS = ((12, 12), (W - 13, 12),
                  (12, H - 13), (W - 13, H - 13))

OBJECT_TEMPLATE = np.zeros((1, A.cols * A.rows, 3), dtype=np.float64)
OBJECT_TEMPLATE[0, :, :2] = (
    np.mgrid[0:A.cols, 0:A.rows].T.reshape(-1, 2) * A.square_size)
CURRENT_K = np.array([[F_GUESS, 0.0, CX],
                      [0.0, F_GUESS, CY],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
CURRENT_D = np.zeros((4, 1), dtype=np.float64)
if A.initial_intrinsics:
    with np.load(A.initial_intrinsics) as initial:
        CURRENT_K = np.asarray(initial[A.camera + "_K"], dtype=np.float64)
        CURRENT_D = np.asarray(initial[A.camera + "_D"], dtype=np.float64).reshape(4, 1)

CAMERA_R = CAMERA_T = None
if A.extrinsics:
    with np.load(A.extrinsics) as fixed:
        CAMERA_R = np.asarray(
            fixed[A.camera + "_R_camera_to_vehicle"], dtype=np.float64)
        CAMERA_T = np.asarray(
            fixed[A.camera + "_t_vehicle_m"], dtype=np.float64).reshape(3)

LIVE_NPZ = os.path.join(OUT, "live_%s_calibration.npz" % A.camera)
LIVE_JSON = LIVE_NPZ + ".json"
OBSERVATIONS = []
POSES = []

STATE = {
    "views": 0, "max_angle": 0.0, "cells": np.zeros((N_RING, N_SECT), dtype=int),
    "found": False, "hint": "Connecting to the car...", "level": "wait",
    "fps": 0.0, "last_jpeg": None, "sharp": 0.0, "target": None, "done": False,
    "frame_seq": 0, "last_frame_time": 0.0,
    "corner_angles": np.zeros(4, dtype=float), "angle_target": None,
    "calibration_status": "initial matrix; collecting live views",
    "calibration_rms": None, "calibration_views": 0,
    "calibration_output": LIVE_NPZ,
}
LOCK = threading.Lock()


def cell_of(px, py):
    dx, dy = px - CX, py - CY
    th = np.degrees(np.hypot(dx, dy) / F_GUESS)
    sec = int(((np.degrees(np.arctan2(dy, dx)) + 360) % 360) / (360.0 / N_SECT))
    ring = np.searchsorted(RING_EDGES, th, side="right") - 1
    return int(np.clip(ring, 0, N_RING - 1)), int(sec), float(th)


def reachable_zones():
    """Return zones containing real pixels in the rectangular image."""
    reachable = np.zeros((N_RING, N_SECT), dtype=bool)
    sums = np.zeros((N_RING, N_SECT, 2), dtype=np.float64)
    counts = np.zeros((N_RING, N_SECT), dtype=int)
    for py in range(0, H, 2):
        for px in range(0, W, 2):
            ring, sec, _ = cell_of(px, py)
            reachable[ring, sec] = True
            sums[ring, sec] += (px, py)
            counts[ring, sec] += 1
    targets = {}
    for ring, sec in zip(*np.where(reachable)):
        targets[(int(ring), int(sec))] = tuple(
            np.rint(sums[ring, sec] / counts[ring, sec]).astype(int))
    return reachable, targets


REACHABLE, TARGET_POINTS = reachable_zones()


def sector_compass(sec):
    """Screen direction of a sector index, as the operator sees it."""
    ang = (sec + 0.5) * (360.0 / N_SECT)
    names = [(0, "RIGHT"), (45, "LOWER RIGHT"), (90, "DOWN"), (135, "LOWER LEFT"),
             (180, "LEFT"), (225, "UPPER LEFT"), (270, "UP"), (315, "UPPER RIGHT")]
    return min(names, key=lambda t: min(abs(ang - t[0]), 360 - abs(ang - t[0])))[1]


def pick_target():
    c = STATE["cells"]
    for ring in range(N_RING - 1, -1, -1):          # outermost first: it matters most
        empty = [s for s in range(N_SECT)
                 if REACHABLE[ring, s] and
                 c[ring, s] < MIN_VIEWS_PER_ZONE]
        if empty:
            return ring, empty[len(empty) // 2]
    return None


def pick_angle_target(current=None):
    if (current is not None and
            STATE["corner_angles"][current] + ANGLE_TOLERANCE_DEG <
            ANGLE_GOALS_DEG[current]):
        return current
    missing = [index for index, angle in enumerate(STATE["corner_angles"])
               if angle + ANGLE_TOLERANCE_DEG < ANGLE_GOALS_DEG[index]]
    return missing[0] if missing else None


def corner_index(px, py):
    return (1 if px >= CX else 0) + (2 if py >= CY else 0)


class Puller(threading.Thread):
    daemon = True

    def __init__(self):
        super().__init__()
        self.last_save_time = 0.0
        self.last_saved_corners = None

    def run(self):
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        while True:
            try:
                s = socket.create_connection((A.car, A.car_port), timeout=10)
                s.settimeout(8.0)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                with LOCK:
                    STATE["hint"] = "Connected. Hold the board facing the car."
                t_prev = time.time()
                while True:
                    s.sendall(b"\x01")
                    head = b""
                    while len(head) < 4:
                        b = s.recv(4 - len(head))
                        if not b:
                            raise IOError("closed")
                        head += b
                    n = struct.unpack("!I", head)[0]
                    data = b""
                    while len(data) < n:
                        b = s.recv(min(65536, n - len(data)))
                        if not b:
                            raise IOError("closed")
                        data += b
                    if n == 0:
                        continue
                    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    self.process(img, crit)
                    now = time.time()
                    with LOCK:
                        STATE["fps"] = 0.7 * STATE["fps"] + 0.3 / max(now - t_prev, 1e-3)
                        STATE["frame_seq"] += 1
                        STATE["last_frame_time"] = now
                    t_prev = now
            except Exception as e:
                with LOCK:
                    STATE["hint"] = "Lost the car (%s). Retrying..." % type(e).__name__
                    STATE["level"] = "bad"
                time.sleep(2)

    def process(self, img, crit):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ok, cor = cv2.findChessboardCorners(
            g, (A.cols, A.rows),
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)
        vis = img.copy()
        sharp = float(cv2.Laplacian(g, cv2.CV_64F).var())

        with LOCK:
            tgt = STATE["target"] or pick_target()
            STATE["target"] = tgt
            angle_tgt = (None if tgt is not None else
                         pick_angle_target(STATE["angle_target"]))
            STATE["angle_target"] = angle_tgt

        if ok:
            cv2.cornerSubPix(g, cor, (5, 5), (-1, -1), crit)
            pts = cor.reshape(-1, 2)
            cv2.drawChessboardCorners(vis, (A.cols, A.rows), cor, ok)
            cxp, cyp = float(pts[:, 0].mean()), float(pts[:, 1].mean())
            _, _, th = cell_of(cxp, cyp)
            observed_cells = sorted(set(
                cell_of(float(px), float(py))[:2] for px, py in pts))
            observed_corner_angles = np.zeros(4, dtype=float)
            for px, py in pts:
                _, _, angle = cell_of(float(px), float(py))
                index = corner_index(float(px), float(py))
                observed_corner_angles[index] = max(
                    observed_corner_angles[index], angle)
            with LOCK:
                fresh_cells = [cell for cell in observed_cells
                               if REACHABLE[cell] and
                               STATE["cells"][cell] < MIN_VIEWS_PER_ZONE]
                angle_improvements = [
                    index for index, angle in enumerate(observed_corner_angles)
                    if (angle > STATE["corner_angles"][index] + 0.5 and
                        STATE["corner_angles"][index] + ANGLE_TOLERANCE_DEG <
                        ANGLE_GOALS_DEG[index])]
                fresh = bool(fresh_cells or angle_improvements)
                blurry = sharp < 25.0
                now = time.time()
                enough_time = now - self.last_save_time >= A.capture_interval
                novelty = (float("inf") if self.last_saved_corners is None else
                           float(np.linalg.norm(
                               pts - self.last_saved_corners, axis=1).mean()))
                novel = novelty >= A.novelty_px
                if blurry:
                    STATE["hint"] = "Too blurry - hold still"; STATE["level"] = "warn"
                elif fresh and enough_time and novel:
                    for cell in set(observed_cells):
                        if REACHABLE[cell]:
                            STATE["cells"][cell] += 1
                    STATE["corner_angles"] = np.maximum(
                        STATE["corner_angles"], observed_corner_angles)
                    STATE["views"] += 1
                    STATE["max_angle"] = max(STATE["max_angle"], float(np.degrees(
                        np.hypot(pts[:, 0] - CX, pts[:, 1] - CY).max() / F_GUESS)))
                    cv2.imwrite(os.path.join(OUT, "v%05d.png" % STATE["views"]), img)
                    self.last_save_time = now
                    self.last_saved_corners = pts.copy()
                    STATE["hint"] = "CAPTURED  (%d deg)" % th; STATE["level"] = "good"
                    STATE["target"] = pick_target()
                    STATE["angle_target"] = (
                        None if STATE["target"] is not None
                        else pick_angle_target(STATE["angle_target"]))
                elif fresh:
                    STATE["hint"] = "Detected - move or tilt the board"
                    STATE["level"] = "wait"
                else:
                    t = STATE["target"]
                    angle_t = STATE["angle_target"]
                    if t is None and angle_t is None:
                        STATE["hint"] = "ALL ZONES COVERED - you can stop"
                        STATE["level"] = "good"; STATE["done"] = True
                    elif t is None:
                        STATE["hint"] = (
                            "Zones done - put one INNER corner at %s (goal %.0f deg)" %
                            (CORNER_NAMES[angle_t], ANGLE_GOALS_DEG[angle_t]))
                        STATE["level"] = "warn"
                    else:
                        STATE["hint"] = "Covered here - move %s, further out" % sector_compass(t[1])
                        STATE["level"] = "warn"
                STATE["found"] = True
        else:
            with LOCK:
                STATE["found"] = False
                t = STATE["target"]
                angle_t = STATE["angle_target"]
                if t is not None:
                    STATE["hint"] = "Board not detected - aim %s" % sector_compass(t[1])
                elif angle_t is not None:
                    STATE["hint"] = "Put one INNER corner at %s" % CORNER_NAMES[angle_t]
                else:
                    STATE["hint"] = "Board not detected - show the CHECKER side, whole board in frame"
                STATE["level"] = "bad"

        self.overlay(vis)
        okj, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        with LOCK:
            STATE["sharp"] = sharp
            if okj:
                STATE["last_jpeg"] = buf.tobytes()

    def overlay(self, vis):
        with LOCK:
            cells = STATE["cells"].copy(); tgt = STATE["target"]
            angle_tgt = STATE["angle_target"]
        for ring in range(N_RING):
            r1 = int(np.radians(RING_EDGES[ring + 1]) * F_GUESS)
            for sec in range(N_SECT):
                if not REACHABLE[ring, sec]:
                    continue
                a0, a1 = sec * (360.0 / N_SECT), (sec + 1) * (360.0 / N_SECT)
                col = ((60, 190, 60)
                       if cells[ring, sec] >= MIN_VIEWS_PER_ZONE
                       else (55, 55, 210))
                if tgt == (ring, sec):
                    col = (0, 225, 255)
                thickness = 5 if tgt == (ring, sec) else 2
                cv2.ellipse(vis, (int(CX), int(CY)), (r1, r1),
                            0, a0, a1, col, thickness)
        if tgt is not None:
            p = TARGET_POINTS[tgt]
            cv2.arrowedLine(vis, (int(CX), int(CY)), p, (0, 225, 255), 3, tipLength=0.15)
            cv2.circle(vis, p, 26, (0, 225, 255), 3)
        elif angle_tgt is not None:
            p = CORNER_TARGETS[angle_tgt]
            cv2.arrowedLine(vis, (int(CX), int(CY)), p,
                            (0, 225, 255), 4, tipLength=0.12)
            cv2.circle(vis, p, 18, (0, 225, 255), 4)


PAGE = b"""<!doctype html><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>QCar calibration</title>
<style>
:root{--bg:#10161c;--fg:#eef3f6;--mut:#8b9aa5;--good:#2fae62;--warn:#e0a32e;--bad:#d0493c}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.4 -apple-system,Segoe UI,Roboto,sans-serif;-webkit-user-select:none}
header{padding:10px 14px;border-bottom:1px solid #22303a;display:flex;
justify-content:space-between;align-items:center}
h1{font-size:15px;margin:0;letter-spacing:.04em}
#cam{width:100%;display:block;background:#000}
#hint{padding:16px 14px;font-size:22px;font-weight:700;text-align:center;line-height:1.25}
.good{background:var(--good)}.warn{background:var(--warn);color:#241a00}
.bad{background:var(--bad)}.wait{background:#2a3640}
#bars{padding:12px 14px}
.row{display:flex;justify-content:space-between;font-size:13px;color:var(--mut);margin:9px 0 3px}
.bar{height:9px;background:#1d2830;border-radius:5px;overflow:hidden}
.fill{height:100%;background:var(--good);transition:width .3s}
.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:2px;padding:0 14px 16px}
.c{aspect-ratio:1;border-radius:2px;background:#46515a}
.c.reach{background:#d0493c}
.c.on{background:#2fae62}.c.t{background:#00e1ff}
small{color:var(--mut);padding:0 14px 20px;display:block}
</style>
<header><h1>QCAR CALIBRATION &mdash; <span id=cam_name></span></h1><span id=fps style="color:#8b9aa5;font-size:12px"></span></header>
<img id=cam src="/stream?t=0">
<div id=hint class=wait>...</div>
<div id=bars>
  <div class=row><span>Angular reach</span><span id=angtxt>0&deg; / 70&deg;</span></div>
  <div class=bar><div class=fill id=angbar style="width:0%"></div></div>
  <div class=row><span>Four corners</span><span id=cornertxt>0&deg; / 0&deg; / 0&deg; / 0&deg;</span></div>
  <div class=row><span>Zones covered</span><span id=zonetxt>0 / 48</span></div>
  <div class=bar><div class=fill id=zonebar style="width:0%"></div></div>
</div>
<div class=grid id=grid></div>
<small>Green = covered &middot; Red = still missing &middot; Cyan = go here now.
Keep the whole board visible, checker side toward the car.</small>
<script>
const grid=document.getElementById('grid');
let cells=[];for(let i=0;i<48;i++){const d=document.createElement('div');d.className='c';grid.appendChild(d);cells.push(d);}
let lastSeq=-1,lastChange=Date.now();
async function tick(){
 try{const r=await fetch('/state');const s=await r.json();
  const h=document.getElementById('hint');h.textContent=s.hint;h.className=s.level;
  document.getElementById('cam_name').textContent=s.camera.toUpperCase();
  document.getElementById('fps').textContent=s.fps.toFixed(1)+' fps';
  document.getElementById('angtxt').textContent=s.max_angle.toFixed(0)+'\\u00b0 / 70\\u00b0';
  document.getElementById('angbar').style.width=Math.min(100,s.max_angle/70*100)+'%';
  document.getElementById('cornertxt').textContent=
   s.corner_angles.map(v=>v.toFixed(0)+'\\u00b0').join(' / ')+
   ' (goals '+s.angle_goals.map(v=>v.toFixed(0)+'\\u00b0').join(' / ')+')';
  let n=0,total=0;s.cells.forEach((row,ri)=>row.forEach((v,si)=>{
   if(s.reachable[ri][si]){total++;if(v>=s.min_views_per_zone)n++;}}));
  document.getElementById('zonetxt').textContent=n+' / '+total;
  document.getElementById('zonebar').style.width=(n/total*100)+'%';
  s.cells.forEach((row,ri)=>row.forEach((v,si)=>{const i=ri*12+si;
   cells[i].className='c'+(s.reachable[ri][si]?' reach':'')+
    (v>=s.min_views_per_zone?' on':'')+
    ((s.target&&s.target[0]==ri&&s.target[1]==si)?' t':'');}));
  if(s.frame_seq!==lastSeq){lastSeq=s.frame_seq;lastChange=Date.now();}
  if(Date.now()-lastChange>4000){
   document.getElementById('cam').src='/stream?t='+Date.now();lastChange=Date.now();}
 }catch(e){}
 setTimeout(tick,400);}
tick();
</script>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(PAGE)))
            self.end_headers(); self.wfile.write(PAGE)
        elif path == "/state":
            with LOCK:
                body = json.dumps({
                    "views": STATE["views"], "max_angle": STATE["max_angle"],
                    "cells": STATE["cells"].tolist(), "hint": STATE["hint"],
                    "level": STATE["level"], "fps": STATE["fps"],
                    "target": list(STATE["target"]) if STATE["target"] else None,
                    "camera": A.camera, "done": STATE["done"],
                    "frame_seq": STATE["frame_seq"],
                    "reachable": REACHABLE.tolist(),
                    "min_views_per_zone": MIN_VIEWS_PER_ZONE,
                    "corner_angles": STATE["corner_angles"].tolist(),
                    "angle_target": STATE["angle_target"],
                    "angle_goals": ANGLE_GOALS_DEG.tolist(),
                }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)
        elif path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.end_headers()
            try:
                while True:
                    with LOCK:
                        j = STATE["last_jpeg"]
                    if j:
                        self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(j)).encode() + b"\r\n\r\n" + j + b"\r\n")
                        self.wfile.flush()
                    time.sleep(0.08)
            except Exception:
                pass
        else:
            self.send_response(404); self.end_headers()


if A.seed_state and os.path.isfile(A.seed_state):
    with open(A.seed_state) as stream:
        seed_state = json.load(stream)
    seed_cells = np.asarray(seed_state["cells"], dtype=int)
    if seed_cells.shape != (N_RING, N_SECT) or np.any(seed_cells < 0):
        raise SystemExit("invalid cells in --seed-state")
    STATE["cells"][:] = seed_cells
    STATE["views"] = int(seed_state["views"])
    STATE["max_angle"] = float(seed_state["max_angle"])
    if "corner_angles" in seed_state:
        STATE["corner_angles"][:] = np.asarray(
            seed_state["corner_angles"], dtype=float)
    print("seeded %d earlier views, reach %.1f deg from %s" %
          (STATE["views"], STATE["max_angle"], A.seed_state))
elif A.seed and os.path.isdir(A.seed):
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    import glob
    for p in sorted(glob.glob(os.path.join(A.seed, "*.png"))):
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if g is None: continue
        ok, cor = cv2.findChessboardCorners(g, (A.cols, A.rows),
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not ok: continue
        pts = cor.reshape(-1, 2)
        for cell in set(cell_of(float(px), float(py))[:2]
                        for px, py in pts):
            if REACHABLE[cell]:
                STATE["cells"][cell] += 1
        STATE["views"] += 1
        STATE["max_angle"] = max(STATE["max_angle"], float(np.degrees(
            np.hypot(pts[:, 0] - CX, pts[:, 1] - CY).max() / F_GUESS)))
    print("seeded %d earlier views, reach %.1f deg" % (STATE["views"], STATE["max_angle"]))

Puller().start()
print("open  http://%s:%d  on your phone" % (
    socket.gethostbyname(socket.gethostname()), A.http_port))
ThreadingHTTPServer(("0.0.0.0", A.http_port), Handler).serve_forever()
