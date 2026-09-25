"""Live viewer for run.py: three windows -- each CAV's point of view (the
undistorted front image HEAL actually sees, with the fused detections
projected into it) and a bird's-eye view of the lab (walls, both cars' Vicon
poses, detections in world metres).

Drawing is done with OpenCV on numpy images; the windows are plain Tkinter
(+ PIL ImageTk), because the heal38 env ships opencv-python-headless (no
cv2.imshow) and swapping OpenCV builds inside the training env is not worth
the risk. matplotlib was tried first: ~107 ms per redraw, too slow for 10 Hz.

Box projection reuses qcar/visualize.py's project/draw code and
patch_real_extrinsic's camera geometry -- the same geometry the model's LSS
encoder uses -- so a box drawn on a POV is where the model "thinks" it is.
Detections are in the ego's frame; for the peer's POV they are moved into
the peer's frame through the two Vicon poses.
"""
import math
import time

import cv2
import numpy as np

EGO_COLOR = (80, 200, 60)      # BGR green
PEER_COLOR = (230, 140, 40)    # BGR blue-ish
DET_COLOR = (255, 0, 255)      # magenta, same as qcar/visualize.py predictions
QCAR_LENGTH, QCAR_WIDTH = 0.42, 0.19   # metres, for drawing only


def _rot(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def ego_model_to_world_m(xy_model, ego_pose):
    """(N,2) ego-frame x10 model units -> (N,2) Vicon world metres."""
    return (xy_model / 10.0) @ _rot(ego_pose["yaw"]).T + [ego_pose["x"], ego_pose["y"]]


def world_m_to_cav_model(xy_world, cav_pose):
    return ((xy_world - [cav_pose["x"], cav_pose["y"]]) @ _rot(cav_pose["yaw"])) * 10.0


class LiveViewer:
    def __init__(self, agents, ego_id, car_labels, cam_K, walls, bev_range, max_hz):
        import tkinter as tk
        from PIL import Image, ImageTk
        from qcar.patches.patch_real_extrinsic import CAMERA_PARAMS
        from qcar.visualize import draw_boxes

        self.Image, self.ImageTk = Image, ImageTk
        self.draw_boxes, self.cam_extr = draw_boxes, CAMERA_PARAMS
        self.agents, self.ego_id, self.labels, self.K = agents, ego_id, car_labels, cam_K
        self.walls, self.range = walls, bev_range
        self.period = 1.0 / float(max_hz)
        self.last_draw = 0.0
        self.last_pump = 0.0
        self.quit = False
        self.scale = 100.0  # px per metre

        self.root = tk.Tk()
        self.root.withdraw()  # only the three Toplevels are shown
        self.labels_tk, self.photos = {}, {}
        titles = dict((a, "POV CAV %s - %s%s" % (a, car_labels[a], " (ego)" if a == ego_id else ""))
                      for a in agents)
        titles["bev"] = "Vista de pajaro (Vicon, metros)"
        for key, title in titles.items():
            win = tk.Toplevel(self.root)
            win.title(title)
            win.protocol("WM_DELETE_WINDOW", self._on_close)
            win.bind("<Key-q>", lambda _e: self._on_close())
            win.bind("<Escape>", lambda _e: self._on_close())
            label = tk.Label(win, bg="black")
            label.pack()
            self.labels_tk[key] = label
        self.waiting("Esperando datos de los carros...")

    def waiting(self, text):
        """Placeholder shown until the first fused pair (Tk windows are
        otherwise blank), and again with `text` updated by run.py."""
        for key in self.labels_tk:
            size = (480, 640) if key != "bev" else (600, 700)
            img = np.full(size + (3,), 40, dtype=np.uint8)
            cv2.putText(img, text, (20, size[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1, cv2.LINE_AA)
            self._show(key, img)
        try:
            self.root.update()
        except Exception:  # noqa: BLE001 -- window torn down
            self.quit = True

    def _on_close(self):
        self.quit = True

    def pump(self):
        """Keep the windows responsive while waiting for pairs."""
        now = time.time()
        if now - self.last_pump >= 0.03:
            try:
                self.root.update()
            except Exception:  # noqa: BLE001 -- window torn down
                self.quit = True
            self.last_pump = now

    def due(self):
        return time.time() - self.last_draw >= self.period

    # --------------------------------------------------------------- drawing
    def _show(self, key, bgr):
        photo = self.ImageTk.PhotoImage(self.Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        self.labels_tk[key].configure(image=photo)
        self.photos[key] = photo  # Tk keeps no reference of its own

    def _pov(self, aid, frame, boxes_ego, frames):
        img = frame["front_undist"].copy()
        n_drawn = 0
        if len(boxes_ego):
            boxes = boxes_ego.copy()
            if aid != self.ego_id:
                world = ego_model_to_world_m(boxes[:, :, :2].reshape(-1, 2),
                                             frames[self.ego_id]["ego_pose"])
                boxes[:, :, :2] = world_m_to_cav_model(world, frame["ego_pose"]).reshape(-1, 8, 2)
            R_cb, t_cb = self.cam_extr[aid]
            for b in boxes:
                ctx = {"im": img, "K": self.K[aid], "R_cb": R_cb, "t_cb": t_cb}
                n_drawn += bool(self.draw_boxes(ctx, b[None], DET_COLOR))
        color = EGO_COLOR if aid == self.ego_id else PEER_COLOR
        cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(img, "CAV %s %s | detecciones visibles aqui: %d"
                    % (aid, "(ego)" if aid == self.ego_id else "(peer)", n_drawn),
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return img

    def _px(self, xy):
        xmin, xmax, ymin, ymax = self.range
        return (int(round((xy[0] - xmin) * self.scale)), int(round((ymax - xy[1]) * self.scale)))

    def _bev(self, frames, boxes_ego, scores, info):
        xmin, xmax, ymin, ymax = self.range
        w, h = int((xmax - xmin) * self.scale), int((ymax - ymin) * self.scale)
        img = np.full((h, w, 3), 255, dtype=np.uint8)
        for gx in np.arange(math.ceil(xmin * 2) / 2, xmax, 0.5):
            cv2.line(img, self._px((gx, ymin)), self._px((gx, ymax)), (235, 235, 235), 1)
        for gy in np.arange(math.ceil(ymin * 2) / 2, ymax, 0.5):
            cv2.line(img, self._px((xmin, gy)), self._px((xmax, gy)), (235, 235, 235), 1)
        cv2.line(img, self._px((0, ymin)), self._px((0, ymax)), (200, 200, 200), 1)
        cv2.line(img, self._px((xmin, 0)), self._px((xmax, 0)), (200, 200, 200), 1)
        for wall in self.walls:
            cv2.line(img, self._px(wall["a"]), self._px(wall["b"]), (60, 60, 60), 4)

        for aid in self.agents:
            pose = frames[aid]["ego_pose"]
            color = EGO_COLOR if aid == self.ego_id else PEER_COLOR
            half = np.array([[QCAR_LENGTH, QCAR_WIDTH], [QCAR_LENGTH, -QCAR_WIDTH],
                             [-QCAR_LENGTH, -QCAR_WIDTH], [-QCAR_LENGTH, QCAR_WIDTH]]) / 2
            poly = half @ _rot(pose["yaw"]).T + [pose["x"], pose["y"]]
            cv2.fillPoly(img, [np.array([self._px(p) for p in poly], dtype=np.int32)], color)
            nose = np.array([pose["x"], pose["y"]]) + 0.35 * np.array(
                [math.cos(pose["yaw"]), math.sin(pose["yaw"])])
            cv2.arrowedLine(img, self._px((pose["x"], pose["y"])), self._px(nose), color, 2,
                            tipLength=0.3)
            cv2.putText(img, "CAV %s%s" % (aid, " ego" if aid == self.ego_id else ""),
                        self._px((pose["x"] + 0.12, pose["y"] + 0.18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        ego_pose = frames[self.ego_id]["ego_pose"]
        for b, s in zip(boxes_ego, scores):
            world = ego_model_to_world_m(b[:4, :2], ego_pose)
            pts = np.array([self._px(p) for p in world], dtype=np.int32)
            cv2.polylines(img, [pts], True, DET_COLOR, 2)
            cv2.putText(img, "%.2f" % s, tuple(pts.max(axis=0) + [3, 0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, DET_COLOR, 1, cv2.LINE_AA)

        lines = ["par %d | agentes fusionados: %s | detecciones: %d"
                 % (info["pair"], info.get("n_agents_fused", "-"), len(scores)),
                 info.get("status", "")]
        for i, text in enumerate(l for l in lines if l):
            cv2.putText(img, text, (10, 22 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 0, 0), 1, cv2.LINE_AA)
        return img

    def update(self, frames, boxes_ego, scores, info):
        """frames: {agent: frame with 'front_undist' and 'ego_pose'};
        boxes_ego: (N,8,3) HEAL output (ego frame, x10); scores: (N,)."""
        for aid in self.agents:
            self._show(aid, self._pov(aid, frames[aid], boxes_ego, frames))
        self._show("bev", self._bev(frames, boxes_ego, scores, info))
        try:
            self.root.update()
        except Exception:  # noqa: BLE001 -- window torn down
            self.quit = True
        self.last_draw = time.time()
