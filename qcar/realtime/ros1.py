"""Minimal pure-Python ROS1 subscriber (XML-RPC master lookup + TCPROS).

The workstation has no ROS install, and each QCar runs its OWN roscore, so a
single rospy process could only ever talk to one car. This talks to any
number of masters from one process, with no ROS dependency: the master and
the publishing node are asked over XML-RPC where a topic lives, then the
message stream is read over the TCPROS wire protocol (4-byte little-endian
length + serialized message). Messages are handed out RAW (serialized bytes);
decoding is done by the caller with rosbags' typestore -- the same decoder
bag_to_dataset_rosbags.py uses, so a live message and a bagged one are
decoded by identical code.

Only what a subscriber needs is implemented: no publishing, no services, no
publisherUpdate callbacks (a lost connection is simply re-resolved).
"""
import os
import socket
import struct
import threading
import time
import xmlrpc.client
from urllib.parse import urlparse


def _encode_header(fields):
    body = b""
    for key, value in fields.items():
        item = ("%s=%s" % (key, value)).encode("utf-8")
        body += struct.pack("<I", len(item)) + item
    return struct.pack("<I", len(body)) + body


def _recv_exact(sock, count):
    chunks, remaining = [], count
    while remaining:
        chunk = sock.recv(min(remaining, 1 << 20))
        if not chunk:
            raise ConnectionError("TCPROS connection closed by publisher")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _decode_header(blob):
    fields, offset = {}, 0
    while offset < len(blob):
        (size,) = struct.unpack_from("<I", blob, offset)
        offset += 4
        key, _, value = blob[offset:offset + size].decode("utf-8").partition("=")
        fields[key] = value
        offset += size
    return fields


def ros1_to_rosbags_type(ros_type):
    """'sensor_msgs/Image' (what a ROS1 master reports) ->
    'sensor_msgs/msg/Image' (what rosbags' typestore expects)."""
    package, _, name = ros_type.partition("/")
    return "%s/msg/%s" % (package, name)


class TopicSubscriber(threading.Thread):
    """Background thread: resolve `topic` on `master_uri`, stream it, call
    `callback(raw_bytes, msgtype, arrival_wall_time)` per message, and
    reconnect forever on any failure until stop() is called."""

    def __init__(self, master_uri, topic, callback, reconnect_sec, log=print):
        threading.Thread.__init__(self, daemon=True)
        self.master_uri = master_uri
        self.topic = topic
        self.callback = callback
        self.reconnect_sec = reconnect_sec
        self.log = log
        self.caller_id = "/heal_realtime_%d" % os.getpid()
        self.messages = 0
        self.bytes = 0
        self._stop = threading.Event()
        self._sock = None

    def stop(self):
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def _resolve(self):
        master = xmlrpc.client.ServerProxy(self.master_uri)
        code, msg, types = master.getTopicTypes(self.caller_id)
        if code != 1:
            raise RuntimeError("getTopicTypes: %s" % msg)
        ros_type = dict(types).get(self.topic)
        if ros_type is None:
            raise LookupError("topic %s not advertised on %s" % (self.topic, self.master_uri))
        code, msg, state = master.getSystemState(self.caller_id)
        if code != 1:
            raise RuntimeError("getSystemState: %s" % msg)
        nodes = dict((t, n) for t, n in state[0]).get(self.topic, [])
        if not nodes:
            raise LookupError("no publisher for %s on %s" % (self.topic, self.master_uri))
        code, msg, node_uri = master.lookupNode(self.caller_id, nodes[0])
        if code != 1:
            raise RuntimeError("lookupNode(%s): %s" % (nodes[0], msg))
        code, msg, proto = xmlrpc.client.ServerProxy(node_uri).requestTopic(
            self.caller_id, self.topic, [["TCPROS"]])
        if code != 1 or not proto:
            raise RuntimeError("requestTopic(%s): %s" % (self.topic, msg))
        _, host, port = proto[0], proto[1], int(proto[2])
        try:
            socket.getaddrinfo(host, port)
        except socket.gaierror:
            # The car advertises its own hostname (ROS_HOSTNAME unset) and
            # this workstation cannot resolve it: the node is on the same
            # machine as the master, so the master's address reaches it.
            host = urlparse(self.master_uri).hostname
        return ros_type, host, port

    def _stream(self):
        ros_type, host, port = self._resolve()
        msgtype = ros1_to_rosbags_type(ros_type)
        sock = socket.create_connection((host, port), timeout=10.0)
        self._sock = sock
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.sendall(_encode_header({
            "callerid": self.caller_id, "topic": self.topic,
            "type": ros_type, "md5sum": "*", "tcp_nodelay": "1",
        }))
        (size,) = struct.unpack("<I", _recv_exact(sock, 4))
        reply = _decode_header(_recv_exact(sock, size))
        if "error" in reply:
            raise RuntimeError("publisher refused %s: %s" % (self.topic, reply["error"]))
        sock.settimeout(None)
        self.log("[ros1] %s <- %s:%d (%s)" % (self.topic, host, port, ros_type))
        while not self._stop.is_set():
            (size,) = struct.unpack("<I", _recv_exact(sock, 4))
            raw = _recv_exact(sock, size)
            arrival = time.time()
            self.messages += 1
            self.bytes += size + 4
            self.callback(raw, msgtype, arrival)

    def run(self):
        while not self._stop.is_set():
            try:
                self._stream()
            except Exception as exc:  # noqa: BLE001 -- any failure means reconnect
                if self._stop.is_set():
                    break
                self.log("[ros1] %s @ %s: %s -- retry in %.1fs"
                         % (self.topic, self.master_uri, exc, self.reconnect_sec))
            finally:
                if self._sock is not None:
                    self._sock.close()
                    self._sock = None
            self._stop.wait(self.reconnect_sec)
