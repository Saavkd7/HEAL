"""Runs ON the QCar and serves fresh JPEG frames from one CSI camera.

Opening a camera is the slow part (PAL needs warm-up reads and the nvargus
daemon dislikes repeated open/close), so this holds the requested camera open
for the whole session and answers one JPEG per request over a TCP socket.

Do not open all four cameras here.  ``QCarCameras.readAll()`` visits every
enabled CSI device, so one stalled secondary camera can freeze a front-camera
calibration.  PAL's ``Camera2D.read()`` also returns whether a new frame was
actually acquired; an unsuccessful read must not resend the old image buffer.

    sudo python3 car_frame_server.py --port 55700 --camera front
"""
from __future__ import print_function

import argparse
import json
import os
import socket
import struct
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))  # only used by _Conf.path_of
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

# Quanser's python libs on the car -- their location is a conf key.
for _p in _conf["frame_server_python_paths"]:
    sys.path.insert(0, _p)
import cv2  # noqa: E402
from pal.products.qcar import QCarCameras  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--conf", default=DEFAULT_CONF,
                 help="Path to the conf.json to read defaults from (default: %s)" % DEFAULT_CONF)
ap.add_argument("--port", type=int, default=_conf["frame_server_port"])
ap.add_argument("--camera", default=_conf["frame_server_camera"],
                 choices=["front", "left", "right", "back"])
ap.add_argument("--quality", type=int, default=_conf["frame_server_quality"])
a = ap.parse_args()

cams = QCarCameras(
    frameWidth=640,
    frameHeight=480,
    frameRate=30,
    enableFront=a.camera == "front",
    enableLeft=a.camera == "left",
    enableRight=a.camera == "right",
    enableBack=a.camera == "back",
)
camera = {
    "front": cams.csiFront,
    "left": cams.csiLeft,
    "right": cams.csiRight,
    "back": cams.csiBack,
}[a.camera]


def read_fresh(attempts=90):
    """Return a newly acquired non-black frame, never PAL's stale buffer."""
    for _ in range(attempts):
        if camera.read():
            image = camera.imageData
            if image is not None and image.size and float(image.mean()) > 5.0:
                return image
        time.sleep(0.01)
    return None


if read_fresh(attempts=180) is None:
    cams.terminate()
    raise SystemExit("camera %s produced no valid frame during warm-up" % a.camera)

srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("0.0.0.0", a.port)); srv.listen(1)
print("READY port=%d camera=%s" % (a.port, a.camera)); sys.stdout.flush()

try:
    while True:
        conn, address = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        conn.settimeout(15.0)
        print("CLIENT %s:%s" % address); sys.stdout.flush()
        try:
            while True:
                if not conn.recv(1):      # one byte = "send a fresh frame"
                    break
                image = read_fresh()
                if image is None:
                    conn.sendall(struct.pack("!I", 0))
                    continue
                ok, buf = cv2.imencode(
                    ".jpg", image,
                    [int(cv2.IMWRITE_JPEG_QUALITY), a.quality])
                data = buf.tobytes() if ok else b""
                conn.sendall(struct.pack("!I", len(data)))
                if data:
                    conn.sendall(data)
        except (IOError, OSError, socket.error) as error:
            print("CLIENT_CLOSED %s" % type(error).__name__)
            sys.stdout.flush()
        finally:
            conn.close()
finally:
    srv.close()
    cams.terminate()
