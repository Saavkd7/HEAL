#!/usr/bin/env python3
"""Port of the QCar's own static_bag_to_dataset.py (ROS1/Python2.7, uses
`rosbag`) to the pure-Python `rosbags` library, so bag conversion can run on
a machine with no ROS1 install at all.

Logic ported 1:1 from
  nvidia@192.168.1.198:/home/nvidia/ros1/src/qcar_bag_collection/scripts/static_bag_to_dataset.py
(fetched 2026-09-13). Same topics, same causal-bounded-wait synchronization,
same --vicon-pose-key auto resolution, same --trim-to-motion logic, same
output schema (front/back/left/right.png + ego_vicon_pose.json +
timestamp.json + metadata.json). Only the ROS I/O layer changed:
rosbags.rosbag1.Reader + its typestore instead of `import rosbag`.

This does NOT fix the ego/target Vicon mislabeling bug (traced separately to
the upstream Vicon bridge/server) -- it only proves conversion can run here.
Usage

 python bag_to_dataset_rosbags.py        # every default comes from conf.json

Flags are optional one-run overrides of conf.json, e.g.
 python bag_to_dataset_rosbags.py --bag <file.bag> --output <converted_dir> --trim-to-motion

"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from collections import deque

import cv2
import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE)))))  # convert/ -> scripts/ -> pipeline/ -> qcar_dataset/ -> data/ -> qcar_testbed_integration/
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



TOPICS = {
    "/qcar/csi_front": "front",
    "/qcar/csi_back": "back",
    "/qcar/csi_left": "left",
    "/qcar/csi_right": "right",
    "/qcar/vicon": "vicon",
}
CAMERAS = ("front", "back", "left", "right")

TYPESTORE = get_typestore(Stores.ROS1_NOETIC)


def atomic_json(path, value):
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
    os.rename(temporary, path)


def stamp_dict(value):
    seconds = int(math.floor(value))
    nanoseconds = int(round((value - seconds) * 1e9))
    if nanoseconds >= 1000000000:
        seconds += 1
        nanoseconds -= 1000000000
    return {"sec": seconds, "nsec": nanoseconds, "float_sec": value}


def source_stamp(msgtype, msg, bag_stamp):
    if msgtype == "sensor_msgs/msg/Image":
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if stamp > 0:
            return stamp
    elif msgtype == "std_msgs/msg/String":
        try:
            envelope = json.loads(msg.data)
            stamp = float(envelope.get("ros_stamp_sec", 0.0))
            if stamp > 0:
                return stamp
        except (TypeError, ValueError):
            pass
    return bag_stamp


def nearest(entries, reference_stamp, cutoff_arrival, tolerance):
    candidates = [
        entry for entry in entries
        if entry["arrival"] <= cutoff_arrival
        and abs(entry["stamp"] - reference_stamp) <= tolerance
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda entry: (
        abs(entry["stamp"] - reference_stamp), -entry["arrival"]))


def latest_causal(entries, reference_stamp, cutoff_arrival, max_age):
    candidates = [
        entry for entry in entries
        if entry["arrival"] <= cutoff_arrival
        and abs(entry["stamp"] - reference_stamp) <= max_age
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda entry: entry["arrival"])


def valid_vehicle_pose(raw_data, requested_pose_key):
    try:
        envelope = json.loads(raw_data)
        payload = envelope.get("data", envelope)
        pose_key = requested_pose_key
        if pose_key == "auto":
            pose_key = str(envelope.get(
                "vicon_pose_key", "ego")).strip().lower()
        if pose_key not in ("ego", "target"):
            return None, None
        vehicle_pose = payload[pose_key]
        values = (
            float(vehicle_pose["x"]), float(vehicle_pose["y"]),
            float(vehicle_pose["yaw"]))
        if not bool(vehicle_pose.get("valid", False)):
            return None, pose_key
        if any(math.isnan(value) or math.isinf(value) for value in values):
            return None, pose_key
        return vehicle_pose, pose_key
    except (KeyError, TypeError, ValueError):
        return None, None


def image_message_to_bgr(msg):
    """Decode the CSI Image directly, mirroring the original's cv_bridge-free path."""
    encoding = msg.encoding.lower()
    channels_by_encoding = {
        "bgr8": 3, "rgb8": 3, "bgra8": 4, "rgba8": 4, "mono8": 1,
    }
    if encoding not in channels_by_encoding:
        raise ValueError("Unsupported camera encoding: %s" % msg.encoding)

    channels = channels_by_encoding[encoding]
    row_bytes = int(msg.step)
    useful_row_bytes = int(msg.width) * channels
    if row_bytes < useful_row_bytes:
        raise ValueError("Image step is smaller than encoded row width")

    raw = np.frombuffer(msg.data.tobytes() if hasattr(msg.data, "tobytes") else msg.data,
                         dtype=np.uint8)
    required_bytes = int(msg.height) * row_bytes
    if raw.size < required_bytes:
        raise ValueError("Image data is shorter than height * step")
    image = raw[:required_bytes].reshape((int(msg.height), row_bytes))
    image = image[:, :useful_row_bytes].reshape(
        (int(msg.height), int(msg.width), channels))

    if encoding == "bgr8":
        return np.ascontiguousarray(image)
    if encoding == "rgb8":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if encoding == "bgra8":
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if encoding == "rgba8":
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


class StaticBagConverter:
    def __init__(self, args):
        self.args = args
        self.bag_path = os.path.abspath(args.bag)
        self.output = os.path.abspath(args.output)
        start_frame_id = 0
        if os.path.exists(self.output):
            if getattr(args, "add", False):
                # Keep everything already there; new frames continue the
                # numbering instead of colliding with/overwriting 000000.
                existing = [d for d in os.listdir(self.output)
                            if d.isdigit() and os.path.isdir(os.path.join(self.output, d))]
                start_frame_id = (max(int(d) for d in existing) + 1) if existing else 0
            elif getattr(args, "overwrite", False):
                shutil.rmtree(self.output)
                os.makedirs(self.output)
            else:
                raise IOError(
                    "Output directory already exists: %s "
                    "(pass --overwrite to replace it, or --add to append new frames to it)"
                    % self.output)
        else:
            os.makedirs(self.output)

        self.buffers = dict((name, deque()) for name in TOPICS.values())
        self.pending = deque()
        self.raw_first_front_stamp = None
        self.requested_start_stamp = None
        self.requested_end_stamp = None
        self.next_reference_stamp = None
        self.first_reference_stamp = None
        self.last_reference_stamp = None
        self.motion_origin = None
        self.motion_candidate_stamp = None
        self.motion_candidate_count = 0
        self.motion_start_stamp = None
        self.motion_start_displacement = None
        self.resolved_vicon_pose_key = None
        self.references_scheduled = 0
        self.frame_offset = start_frame_id  # --add: continue numbering past what's already there
        self.frames_saved = 0  # frames saved by THIS run only (metadata/fps stay meaningful)
        self.rejections = {}
        self.retention = max(
            args.camera_tolerance, args.vicon_max_age
        ) + args.wait_sec + 1.0

    def reject(self, reason):
        self.rejections[reason] = self.rejections.get(reason, 0) + 1

    def selected_vehicle_pose(self, raw_data):
        pose, pose_key = valid_vehicle_pose(raw_data, self.args.vicon_pose_key)
        if pose_key is not None:
            if self.resolved_vicon_pose_key is None:
                self.resolved_vicon_pose_key = pose_key
            elif pose_key != self.resolved_vicon_pose_key:
                raise RuntimeError(
                    "Vicon pose key changed inside bag from %s to %s" % (
                        self.resolved_vicon_pose_key, pose_key))
        return pose

    def run(self):
        with Reader(self.bag_path) as reader:
            topic_names = set(TOPICS.keys())
            present = {c.topic for c in reader.connections}
            missing = [t for t in topic_names if t not in present]
            if missing:
                raise RuntimeError(
                    "Bag is missing required topics: %s" % ", ".join(missing))

            connections = [c for c in reader.connections if c.topic in topic_names]
            for connection, bag_timestamp_ns, rawdata in reader.messages(
                    connections=connections):
                arrival = bag_timestamp_ns / 1e9
                msg = TYPESTORE.deserialize_ros1(rawdata, connection.msgtype)
                raw_for_json = msg.data if connection.msgtype == "std_msgs/msg/String" else None
                entry = {
                    "stamp": source_stamp(connection.msgtype, msg, arrival),
                    "arrival": arrival,
                    "message": msg,
                    "raw": raw_for_json,
                }
                name = TOPICS[connection.topic]
                self.buffers[name].append(entry)
                if name == "vicon" and self.args.trim_to_motion:
                    self.update_motion_state(entry)
                if name == "front":
                    self.maybe_schedule_reference(entry)
                self.finalize_ready(arrival)
                self.prune(arrival)

            self.finalize_ready(float("inf"))

        self.write_metadata()
        print("Converted %d frames; rejected %d references" % (
            self.frames_saved, sum(self.rejections.values())))
        print("Output: %s" % self.output)
        if self.frames_saved == 0:
            raise RuntimeError(
                "No synchronized frames were exported; inspect metadata.json")

    def update_motion_state(self, entry):
        if self.motion_start_stamp is not None:
            return
        ego_pose = self.selected_vehicle_pose(entry["raw"])
        if ego_pose is None:
            self.motion_candidate_stamp = None
            self.motion_candidate_count = 0
            return

        position = np.asarray(
            [float(ego_pose["x"]), float(ego_pose["y"])], dtype=np.float64)
        if self.motion_origin is None:
            self.motion_origin = position
            return

        displacement = float(np.linalg.norm(position - self.motion_origin))
        if displacement < self.args.motion_threshold_m:
            self.motion_candidate_stamp = None
            self.motion_candidate_count = 0
            return

        if self.motion_candidate_count == 0:
            self.motion_candidate_stamp = entry["stamp"]
        self.motion_candidate_count += 1
        if self.motion_candidate_count < self.args.motion_confirm_samples:
            return

        self.motion_start_stamp = self.motion_candidate_stamp
        self.motion_start_displacement = displacement
        print(
            "Confirmed Vicon motion at %.9f s (displacement %.4f m)" % (
                self.motion_start_stamp, displacement))

    def maybe_schedule_reference(self, entry):
        period = 1.0 / self.args.fps
        if self.raw_first_front_stamp is None:
            self.raw_first_front_stamp = entry["stamp"]

        if self.args.trim_to_motion and self.motion_start_stamp is None:
            return

        if self.requested_start_stamp is None:
            self.requested_start_stamp = (
                self.raw_first_front_stamp + self.args.trim_start_sec)
            if self.args.trim_to_motion:
                self.requested_start_stamp = max(
                    self.requested_start_stamp, self.motion_start_stamp)
            if self.args.duration_sec is not None:
                self.requested_end_stamp = (
                    self.requested_start_stamp + self.args.duration_sec)
            self.next_reference_stamp = self.requested_start_stamp

        if entry["stamp"] < self.next_reference_stamp - 1e-6:
            return
        if (self.requested_end_stamp is not None and
                entry["stamp"] >= self.requested_end_stamp - 1e-6):
            return

        reference = dict(entry)
        reference["cutoff"] = entry["arrival"] + self.args.wait_sec
        self.pending.append(reference)
        self.references_scheduled += 1
        if self.first_reference_stamp is None:
            self.first_reference_stamp = entry["stamp"]

        while self.next_reference_stamp <= entry["stamp"] + 1e-6:
            self.next_reference_stamp += period

    def finalize_ready(self, current_arrival):
        while self.pending and self.pending[0]["cutoff"] <= current_arrival:
            self.convert_reference(self.pending.popleft())

    def selections_for(self, reference):
        stamp = reference["stamp"]
        cutoff = reference["cutoff"]
        selected = {"front": reference}
        for name in ("back", "left", "right"):
            selected[name] = nearest(
                self.buffers[name], stamp, cutoff, self.args.camera_tolerance)
        selected["vicon"] = latest_causal(
            self.buffers["vicon"], stamp, cutoff, self.args.vicon_max_age)
        return selected

    def convert_reference(self, reference):
        selected = self.selections_for(reference)
        for name in ("back", "left", "right", "vicon"):
            if selected[name] is None:
                self.reject("missing_or_stale_" + name)
                return

        ego_pose = self.selected_vehicle_pose(selected["vicon"]["raw"])
        if ego_pose is None:
            self.reject("invalid_vicon_vehicle_pose")
            return

        frame_id = "%06d" % (self.frame_offset + self.frames_saved)
        frame_dir = os.path.join(self.output, frame_id)
        os.makedirs(frame_dir)
        image_options = [cv2.IMWRITE_PNG_COMPRESSION, self.args.png_compression_level]
        for name in CAMERAS:
            image = image_message_to_bgr(selected[name]["message"])
            image_path = os.path.join(frame_dir, name + ".png")
            if not cv2.imwrite(image_path, image, image_options):
                raise IOError("Failed to write %s" % image_path)

        atomic_json(os.path.join(frame_dir, "ego_vicon_pose.json"), ego_pose)

        selected_timing = {}
        for name, selected_entry in selected.items():
            selected_timing[name] = {
                "source_stamp": stamp_dict(selected_entry["stamp"]),
                "bag_arrival_stamp": stamp_dict(selected_entry["arrival"]),
                "source_offset_from_reference_sec": (
                    selected_entry["stamp"] - reference["stamp"]),
                "available_before_decision_sec": (
                    reference["cutoff"] - selected_entry["arrival"]),
            }
        atomic_json(os.path.join(frame_dir, "timestamp.json"), {
            "frame_id": frame_id,
            "reference_source_time_sec": reference["stamp"],
            "reference_bag_time_sec": reference["arrival"],
            "reference_time_sec": reference["stamp"],
            "reference_source_stamp": stamp_dict(reference["stamp"]),
            "reference_arrival_stamp": stamp_dict(reference["arrival"]),
            "decision_cutoff_arrival_stamp": stamp_dict(reference["cutoff"]),
            "bounded_wait_sec": self.args.wait_sec,
            "causal_by_bag_arrival": True,
            "selected": selected_timing,
            "references": {"ego": "ego_vicon_pose.json"},
        })

        self.frames_saved += 1
        self.last_reference_stamp = reference["stamp"]
        if self.frames_saved % 10 == 0 or self.frames_saved == 1:
            print("Saved frame %s" % frame_id)

    def prune(self, current_arrival):
        threshold = current_arrival - self.retention
        for entries in self.buffers.values():
            while entries and entries[0]["arrival"] < threshold:
                entries.popleft()

    def write_metadata(self):
        duration = None
        effective_fps = None
        if (self.first_reference_stamp is not None and
                self.last_reference_stamp is not None):
            duration = self.last_reference_stamp - self.first_reference_stamp
            if duration > 0 and self.frames_saved > 1:
                effective_fps = (self.frames_saved - 1) / duration

        metadata = {
            "source_bag": self.bag_path,
            "dataset_type": "static_four_camera_vicon",
            "converter": "bag_to_dataset_rosbags.py (port of onboard static_bag_to_dataset.py)",
            "topics": sorted(TOPICS.keys()),
            "synchronization_mode": "causal_bounded_wait",
            "target_fps": self.args.fps,
            "bounded_wait_sec": self.args.wait_sec,
            "camera_tolerance_sec": self.args.camera_tolerance,
            "vicon_max_age_sec": self.args.vicon_max_age,
            "requested_vicon_pose_key": self.args.vicon_pose_key,
            "resolved_vicon_pose_key": self.resolved_vicon_pose_key,
            "trim_start_sec": self.args.trim_start_sec,
            "requested_duration_sec": self.args.duration_sec,
            "raw_first_front_source_sec": self.raw_first_front_stamp,
            "requested_start_source_sec": self.requested_start_stamp,
            "requested_end_source_sec": self.requested_end_stamp,
            "trim_to_motion": self.args.trim_to_motion,
            "motion_threshold_m": self.args.motion_threshold_m,
            "motion_confirm_samples": self.args.motion_confirm_samples,
            "motion_start_source_sec": self.motion_start_stamp,
            "motion_start_displacement_m": self.motion_start_displacement,
            "references_scheduled": self.references_scheduled,
            "frames_saved": self.frames_saved,
            "rejections": self.rejections,
            "effective_export_fps": effective_fps,
            "exported_sensor_duration_sec": duration,
            "png_compression_level": self.args.png_compression_level,
        }
        atomic_json(os.path.join(self.output, "metadata.json"), metadata)


def parse_args():
    # First pass just to find --conf (if given), without requiring the other
    # flags yet -- lets --conf point at a non-default conf.json.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--conf", default=DEFAULT_CONF)
    conf_path = pre.parse_known_args()[0].conf
    conf = _load_conf(conf_path)

    parser = argparse.ArgumentParser(
        description=("Synchronize four QCar CSI cameras and local-vehicle Vicon pose "
                     "from a static ROS 1 bag (rosbags port, no ROS install needed). "
                     "Defaults come from conf.json next to this script -- edit that file "
                     "and run with no flags, or pass a flag to override it for one run."))
    parser.add_argument("--conf", default=DEFAULT_CONF,
                         help="Path to the conf.json to read defaults from (default: %s)" % DEFAULT_CONF)
    parser.add_argument("--bag", default=conf.path_of("bag"))
    parser.add_argument("--output", default=conf.path_of("output"))
    parser.add_argument(
        "--overwrite", action="store_true", default=conf["overwrite"],
        help="Delete --output first if it already exists (default: refuse and error out).")
    parser.add_argument(
        "--add", action="store_true", default=conf["add"],
        help="If --output already exists, keep what's there and append this bag's frames, "
             "continuing frame numbering past the highest existing frame id (instead of "
             "refusing, or deleting like --overwrite). Takes precedence over --overwrite.")
    parser.add_argument("--fps", type=float, default=conf["fps"])
    parser.add_argument("--wait-sec", type=float, default=conf["wait_sec"])
    parser.add_argument("--camera-tolerance", type=float, default=conf["camera_tolerance"])
    parser.add_argument("--vicon-max-age", type=float, default=conf["vicon_max_age"])
    parser.add_argument("--vicon-pose-key", choices=("auto", "ego", "target"),
                         default=conf["vicon_pose_key"])
    parser.add_argument("--trim-start-sec", type=float, default=conf["trim_start_sec"])
    parser.add_argument("--duration-sec", type=float, default=conf["duration_sec"])
    parser.add_argument("--trim-to-motion", action="store_true", default=conf["trim_to_motion"])
    parser.add_argument("--motion-threshold-m", type=float, default=conf["motion_threshold_m"])
    parser.add_argument("--motion-confirm-samples", type=int, default=conf["motion_confirm_samples"])
    parser.add_argument("--png-compression-level", type=int, default=conf["png_compression_level"])
    args = parser.parse_args()

    if not args.bag:
        parser.error("--bag not set: pass --bag, or set \"bag\" in %s" % conf_path)
    if not args.output:
        parser.error("--output not set: pass --output, or set \"output\" in %s" % conf_path)
    if args.fps <= 0:
        parser.error("fps must be greater than zero")
    if args.camera_tolerance <= 0 or args.vicon_max_age <= 0:
        parser.error("tolerances must be greater than zero")
    if args.wait_sec < 0 or args.trim_start_sec < 0:
        parser.error("wait-sec and trim-start-sec must be nonnegative")
    if args.duration_sec is not None and args.duration_sec <= 0:
        parser.error("duration-sec must be greater than zero")
    if args.motion_threshold_m <= 0:
        parser.error("motion-threshold-m must be greater than zero")
    if args.motion_confirm_samples < 1:
        parser.error("motion-confirm-samples must be at least one")
    if not 0 <= args.png_compression_level <= 9:
        parser.error("png-compression-level must be 0 through 9")
    return args


def main():
    StaticBagConverter(parse_args()).run()


if __name__ == "__main__":
    main()
