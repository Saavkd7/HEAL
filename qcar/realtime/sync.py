"""Per-car live frame assembly: the causal bounded-wait synchronization of
bag_to_dataset_rosbags.py, driven by live messages instead of a bag.

The selection functions (nearest / latest_causal / valid_vehicle_pose /
source_stamp / image_message_to_bgr / stamp_dict) are IMPORTED from that
script, not copied, so a live frame is assembled by exactly the code that
built the training/validation frames. Only the clock changes: "arrival" is
this workstation's receive time (in the bag it was the car's record time),
and a reference is finalized when the wall clock passes its cutoff instead
of when the bag reader passes it.

Each finalized frame carries the same content the converter writes to disk:
`ego_pose` == ego_vicon_pose.json, `timestamp` == timestamp.json (same keys),
plus the decoded BGR images.
"""
import collections
import json
import threading

import cv2
import numpy as np


class CarSync:
    def __init__(self, conv, cameras, sync_conf, network_lag_sec):
        """conv: the imported bag_to_dataset_rosbags module.
        cameras: camera names to deliver ('front' is always the reference).
        sync_conf: fps / wait_sec / camera_tolerance / vicon_max_age /
        vicon_pose_key, read from the converter's own conf.json.
        network_lag_sec: extra buffer retention. In a bag, a message's
        "arrival" was its record time on the car; live it is the WiFi
        delivery time, and a saturated link delays images far more than the
        small Vicon strings -- without this margin the Vicon sample matching
        a late image has already been pruned (seen live 2026-09-25: ~1 s lag,
        nearly every frame rejected as missing_or_stale_vicon)."""
        self.conv = conv
        self.cameras = ["front"] + [c for c in cameras if c != "front"]
        self.fps = float(sync_conf["fps"])
        self.wait_sec = float(sync_conf["wait_sec"])
        self.camera_tolerance = float(sync_conf["camera_tolerance"])
        self.vicon_max_age = float(sync_conf["vicon_max_age"])
        self.vicon_pose_key = sync_conf["vicon_pose_key"]
        self.retention = (max(self.camera_tolerance, self.vicon_max_age) + self.wait_sec
                          + 1.0 + float(network_lag_sec))
        self.buffers = dict((name, collections.deque()) for name in self.cameras + ["vicon"])
        self.pending = collections.deque()
        self.next_reference_stamp = None
        self.resolved_pose_key = None
        self.rejections = collections.Counter()
        self.frames = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------ input side
    def push(self, name, raw, msgtype, arrival):
        """Called from a subscriber thread for every message of `name`."""
        msg = self.conv.TYPESTORE.deserialize_ros1(raw, msgtype)
        if msgtype == "sensor_msgs/msg/CompressedImage":
            # the converter's source_stamp only knows raw Image; same rule:
            # header stamp when set, else arrival.
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            stamp = stamp if stamp > 0 else arrival
        else:
            stamp = self.conv.source_stamp(msgtype, msg, arrival)
        entry = {"stamp": stamp,
                 "arrival": arrival, "message": msg,
                 "raw": msg.data if msgtype == "std_msgs/msg/String" else None}
        with self._lock:
            self.buffers[name].append(entry)
            if name == "front":
                self._maybe_schedule(entry)

    def _maybe_schedule(self, entry):
        if self.next_reference_stamp is None:
            self.next_reference_stamp = entry["stamp"]
        if entry["stamp"] < self.next_reference_stamp - 1e-6:
            return
        reference = dict(entry)
        reference["cutoff"] = entry["arrival"] + self.wait_sec
        self.pending.append(reference)
        period = 1.0 / self.fps
        while self.next_reference_stamp <= entry["stamp"] + 1e-6:
            self.next_reference_stamp += period

    # ----------------------------------------------------------- output side
    def poll(self, now):
        """Finalize every reference whose bounded wait has expired; return
        the list of assembled frames (oldest first)."""
        out = []
        with self._lock:
            while self.pending and self.pending[0]["cutoff"] <= now:
                frame = self._assemble(self.pending.popleft())
                if frame is not None:
                    out.append(frame)
            threshold = now - self.retention
            for entries in self.buffers.values():
                while entries and entries[0]["arrival"] < threshold:
                    entries.popleft()
        return out

    def _selected_pose(self, raw):
        pose, key = self.conv.valid_vehicle_pose(raw, self.vicon_pose_key)
        if key is not None:
            if self.resolved_pose_key is None:
                self.resolved_pose_key = key
            elif key != self.resolved_pose_key:
                raise RuntimeError("Vicon pose key changed live from %s to %s"
                                   % (self.resolved_pose_key, key))
        return pose

    def _assemble(self, reference):
        stamp, cutoff = reference["stamp"], reference["cutoff"]
        selected = {"front": reference}
        for name in self.cameras[1:]:
            selected[name] = self.conv.nearest(
                self.buffers[name], stamp, cutoff, self.camera_tolerance)
        selected["vicon"] = self.conv.latest_causal(
            self.buffers["vicon"], stamp, cutoff, self.vicon_max_age)
        for name, entry in selected.items():
            if entry is None:
                self.rejections["missing_or_stale_" + name] += 1
                return None
        ego_pose = self._selected_pose(selected["vicon"]["raw"])
        if ego_pose is None:
            self.rejections["invalid_vicon_vehicle_pose"] += 1
            return None

        sd = self.conv.stamp_dict
        timing = {}
        for name, entry in selected.items():
            timing[name] = {
                "source_stamp": sd(entry["stamp"]),
                "workstation_arrival_stamp": sd(entry["arrival"]),
                "source_offset_from_reference_sec": entry["stamp"] - stamp,
                "available_before_decision_sec": cutoff - entry["arrival"],
            }
        timestamp = {
            "frame_id": "%06d" % self.frames,
            "reference_source_time_sec": stamp,
            "reference_time_sec": stamp,
            "reference_source_stamp": sd(stamp),
            "reference_arrival_stamp": sd(reference["arrival"]),
            "decision_cutoff_arrival_stamp": sd(cutoff),
            "bounded_wait_sec": self.wait_sec,
            "causal_by_workstation_arrival": True,
            "selected": timing,
            "references": {"ego": "ego_vicon_pose.json"},
        }
        self.frames += 1
        return {
            "stamp": stamp, "arrival": reference["arrival"],
            "ego_pose": json.loads(json.dumps(ego_pose)),
            "timestamp": timestamp,
            "images": dict((name, selected[name]["message"]) for name in self.cameras),
        }

    def decode_image(self, msg):
        if hasattr(msg, "format"):  # sensor_msgs/CompressedImage (JPEG/PNG)
            data = msg.data.tobytes() if hasattr(msg.data, "tobytes") else bytes(msg.data)
            image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("could not decode CompressedImage (%s)" % msg.format)
            return image
        return self.conv.image_message_to_bgr(msg)
