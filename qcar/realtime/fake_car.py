"""Stand-in for the QCars when they are not on the network: each recorded
.bag is served as if it were that car's live roscore (XML-RPC master + node
+ TCPROS publishers), replaying the bag's messages at their recorded rate.
The bytes on the wire are the bag's own serialized messages, so run.py sees
exactly what a car would send.

    python -m qcar.realtime.fake_car                      # both cars from conf.json
    python -m qcar.realtime.run --master 1=http://127.0.0.1:11411 --master 2=http://127.0.0.1:11412

All bags share one replay clock anchored on the earliest bag start, so the
cars' relative timing is preserved (this is what cross-car pairing needs).
"""
import argparse
import os
import cv2
import numpy as np
import socket
import struct
import threading
import time
from xmlrpc.server import SimpleXMLRPCServer

from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

from qcar import config
from qcar.realtime.ros1 import _decode_header, _encode_header, _recv_exact

HERE = os.path.dirname(os.path.abspath(__file__))
TYPESTORE = get_typestore(Stores.ROS1_NOETIC)


def rosbags_to_ros1_type(msgtype):
    package, _, name = msgtype.partition("/msg/")
    return "%s/%s" % (package, name)


class FakeCar:
    def __init__(self, bag, port, host, image_lag=0.0, jpeg_quality=90):
        self.bag, self.host, self.port = bag, host, port
        self.image_lag, self.jpeg_quality = image_lag, jpeg_quality
        self.node = "/fake_qcar_%d" % port
        with Reader(bag) as r:
            self.start_ns = r.start_time
            self.types = dict((c.topic, (rosbags_to_ros1_type(c.msgtype), c.digest))
                              for c in r.connections)
        # Like `image_transport republish ... compressed` on the car: every
        # raw image topic also exists as <topic>/compressed (JPEG), encoded
        # only while someone subscribes to it.
        for topic, (ros_type, _) in list(self.types.items()):
            if ros_type == "sensor_msgs/Image":
                self.types[topic + "/compressed"] = ("sensor_msgs/CompressedImage", "*")
        self.subscribers = dict((t, []) for t in self.types)
        self.lock = threading.Lock()
        self.tcp = socket.socket()
        self.tcp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.tcp.bind((host, 0))
        self.tcp.listen(16)
        self.uri = "http://%s:%d/" % (host, port)

        rpc = SimpleXMLRPCServer((host, port), logRequests=False, allow_none=True)
        rpc.register_function(lambda cid: [1, "", [[t, ty] for t, (ty, _) in self.types.items()]],
                              "getTopicTypes")
        rpc.register_function(lambda cid: [1, "", [[[t, [self.node]] for t in self.types], [], []]],
                              "getSystemState")
        rpc.register_function(lambda cid, node: [1, "", self.uri], "lookupNode")
        rpc.register_function(
            lambda cid, topic, protos: [1, "", ["TCPROS", host, self.tcp.getsockname()[1]]],
            "requestTopic")
        threading.Thread(target=rpc.serve_forever, daemon=True).start()
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while True:
            conn, _ = self.tcp.accept()
            try:
                (size,) = struct.unpack("<I", _recv_exact(conn, 4))
                topic = _decode_header(_recv_exact(conn, size))["topic"]
                ros_type, md5 = self.types[topic]
                conn.sendall(_encode_header({"callerid": self.node, "topic": topic,
                                             "type": ros_type, "md5sum": md5, "latching": "0"}))
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                with self.lock:
                    self.subscribers[topic].append(conn)
            except Exception as exc:  # noqa: BLE001
                print("[fake_car %d] rejected subscriber: %s" % (self.port, exc))
                conn.close()

    def publish_image(self, topic, raw):
        """Raw image message, optionally delayed by image_lag (a saturated
        link delays the big images, not the small Vicon strings)."""
        def send():
            self.publish(topic, raw)
            compressed = topic + "/compressed"
            if self.subscribers.get(compressed):
                msg = TYPESTORE.deserialize_ros1(raw, "sensor_msgs/msg/Image")
                bgr = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[
                    :, :msg.width * 3].reshape(msg.height, msg.width, 3)
                if msg.encoding.lower() == "rgb8":
                    bgr = cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR)
                ok, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                cmsg = TYPESTORE.types["sensor_msgs/msg/CompressedImage"](
                    header=msg.header, format="bgr8; jpeg compressed bgr8",
                    data=np.frombuffer(jpg.tobytes(), np.uint8))
                self.publish(compressed, TYPESTORE.serialize_ros1(
                    cmsg, "sensor_msgs/msg/CompressedImage"))
        if self.image_lag > 0:
            threading.Timer(self.image_lag, send).start()
        else:
            send()

    def publish(self, topic, raw):
        packet = struct.pack("<I", len(raw)) + bytes(raw)
        with self.lock:
            alive = []
            for conn in self.subscribers[topic]:
                try:
                    conn.sendall(packet)
                    alive.append(conn)
                except OSError:
                    conn.close()
            self.subscribers[topic] = alive


def replay(cars, loop):
    anchor_ns = min(c.start_ns for c in cars)
    while True:
        t0 = time.time()
        threads = []
        for car in cars:
            def run(car=car):
                with Reader(car.bag) as r:
                    for conn, t_ns, raw in r.messages():
                        delay = t0 + (t_ns - anchor_ns) * 1e-9 - time.time()
                        if delay > 0:
                            time.sleep(delay)
                        if car.types[conn.topic][0] == "sensor_msgs/Image":
                            car.publish_image(conn.topic, raw)
                        else:
                            car.publish(conn.topic, raw)
            th = threading.Thread(target=run, daemon=True)
            th.start()
            threads.append(th)
        for th in threads:
            th.join()
        print("[fake_car] replay finished%s" % (", looping" if loop else ""))
        if not loop:
            return


def main():
    rt = config.load_conf(os.path.join(HERE, "conf.json"))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--car", action="append", default=[], metavar="AGENT=BAG",
                    help="override one agent's bag (default: conf fake_car_bags)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--wait_subscribers", type=float, default=0.0,
                    help="seconds to wait before starting the replay")
    ap.add_argument("--no_loop", action="store_true")
    ap.add_argument("--image_lag", type=float, default=0.0,
                    help="delay every image by this many seconds (simulate a saturated WiFi)")
    opt = ap.parse_args()
    bags = dict((a, config.repo_path(b)) for a, b in rt["fake_car_bags"].items())
    for item in opt.car:
        aid, _, bag = item.partition("=")
        bags[aid.strip()] = config.cli_path(bag.strip())
    cars = []
    for aid in sorted(bags):
        car = FakeCar(bags[aid], int(rt["fake_car_ports"][aid]), opt.host, opt.image_lag)
        cars.append(car)
        print("[fake_car] agent %s: %s at %s" % (aid, os.path.basename(bags[aid]), car.uri))
    time.sleep(opt.wait_subscribers)
    replay(cars, loop=rt["fake_car_loop"] and not opt.no_loop)


if __name__ == "__main__":
    main()
