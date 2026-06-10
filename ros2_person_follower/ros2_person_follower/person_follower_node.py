import os
import time
import threading
from typing import List, Tuple

import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist


Box = Tuple[int, int, int, int]  # x, y, w, h


class PersonFollowerNode(Node):
    """Detect the largest person in the camera image and drive toward them."""

    # MobileNet-SSD VOC class id for "person"
    PERSON_CLASS_ID = 15

    def __init__(self) -> None:
        super().__init__("person_follower")

        # ── Parameters ─────────────────────────────────────────────────
        self.declare_parameter("input_topic", "/image")
        self.declare_parameter("output_topic", "/follower/image")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("model_path", "/workspace/models/mobilenet_iter_73000.caffemodel")
        self.declare_parameter("config_path", "/workspace/models/mobilenet_ssd.prototxt")
        self.declare_parameter("confidence_threshold", 0.40)
        self.declare_parameter("max_linear_speed", 0.3)
        self.declare_parameter("max_angular_speed", 0.8)
        self.declare_parameter("target_box_ratio", 0.30)
        self.declare_parameter("dead_zone", 0.05)
        self.declare_parameter("process_fps", 3.0)
        self.declare_parameter("lost_timeout_sec", 2.0)
        self.declare_parameter("enabled", True)

        self.input_topic = self.get_parameter("input_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        self.model_path = self.get_parameter("model_path").value
        self.config_path = self.get_parameter("config_path").value
        self.confidence_threshold = float(self.get_parameter("confidence_threshold").value)
        self.max_linear_speed = float(self.get_parameter("max_linear_speed").value)
        self.max_angular_speed = float(self.get_parameter("max_angular_speed").value)
        # target_box_ratio: desired person bbox width relative to frame width
        # when the box is this big the robot stops (close enough)
        self.target_box_ratio = float(self.get_parameter("target_box_ratio").value)
        self.dead_zone = float(self.get_parameter("dead_zone").value)
        self.process_fps = max(0.5, float(self.get_parameter("process_fps").value))
        self.lost_timeout = float(self.get_parameter("lost_timeout_sec").value)
        self.enabled = bool(self.get_parameter("enabled").value)

        # ── State ──────────────────────────────────────────────────────
        self.bridge = CvBridge()
        self.net = self._load_model()
        self.latest_msg = None
        self.latest_frame_id = 0
        self.last_processed_id = 0
        self.last_person_time = 0.0
        self._lock = threading.Lock()

        # ── Pub / Sub ──────────────────────────────────────────────────
        sub_group = MutuallyExclusiveCallbackGroup()
        timer_group = MutuallyExclusiveCallbackGroup()

        input_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.sub = self.create_subscription(
            Image, self.input_topic, self._on_image, input_qos,
            callback_group=sub_group,
        )
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.img_pub = self.create_publisher(
            Image, self.output_topic,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE),
        )
        self.timer = self.create_timer(
            1.0 / self.process_fps, self._tick, callback_group=timer_group,
        )

        self.get_logger().info(
            "Person follower ready: %s -> %s, speed=%.2f/%.2f, target_ratio=%.0f%%"
            % (self.input_topic, self.cmd_vel_topic,
               self.max_linear_speed, self.max_angular_speed,
               self.target_box_ratio * 100)
        )

    # ── Model ──────────────────────────────────────────────────────────
    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError("Model not found: %s" % self.model_path)
        if not os.path.exists(self.config_path):
            raise FileNotFoundError("Config not found: %s" % self.config_path)
        net = cv2.dnn.readNetFromCaffe(self.config_path, self.model_path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return net

    # ── Callbacks ──────────────────────────────────────────────────────
    def _on_image(self, msg: Image) -> None:
        self.latest_frame_id += 1
        self.latest_msg = msg

    def _tick(self) -> None:
        if not self._lock.acquire(blocking=False):
            return
        try:
            self._process()
        finally:
            self._lock.release()

    def _process(self) -> None:
        msg = self.latest_msg
        if msg is None:
            return
        fid = self.latest_frame_id
        if fid == self.last_processed_id:
            return
        self.last_processed_id = fid

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        h, w = frame.shape[:2]

        # Detect persons
        persons = self._detect_persons(frame)
        now = time.monotonic()

        twist = Twist()

        if persons and self.enabled:
            # Pick the largest person (biggest bounding box area)
            best = max(persons, key=lambda p: p[0][2] * p[0][3])
            box, conf = best
            bx, by, bw, bh = box
            cx = bx + bw / 2.0  # center x of bounding box
            self.last_person_time = now

            # --- Angular: steer toward person center ---
            # error: -1 (person far left) to +1 (person far right)
            error_x = (cx / w) - 0.5  # -0.5 to +0.5
            if abs(error_x) > self.dead_zone:
                # negative because if person is to the right (error>0)
                # we need to turn right (negative angular.z in ROS convention
                # for turning clockwise, but depends on robot setup)
                # Standard: positive angular.z = turn left
                # Person on right = error_x > 0 = turn right = negative angular.z
                twist.angular.z = -error_x * 2.0 * self.max_angular_speed

            # --- Linear: drive forward/backward based on box size ---
            box_ratio = bw / w  # how much of the frame the person fills
            size_error = self.target_box_ratio - box_ratio  # positive = too far
            if abs(size_error) > 0.03:
                twist.linear.x = float(np.clip(
                    size_error * 2.0 * self.max_linear_speed,
                    -self.max_linear_speed,
                    self.max_linear_speed,
                ))

            # Draw tracking info
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            label = "FOLLOW %.0f%%" % (conf * 100)
            cv2.putText(frame, label, (bx, max(by - 8, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            # No person seen
            if self.enabled and (now - self.last_person_time) > self.lost_timeout:
                # Slowly rotate to search
                twist.angular.z = 0.3

        # Draw HUD
        status = "FOLLOWING" if persons and self.enabled else (
            "SEARCHING" if self.enabled else "DISABLED"
        )
        color = (0, 255, 0) if persons and self.enabled else (0, 140, 255)
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        cv2.putText(frame, "v:%.2f w:%.2f" % (twist.linear.x, twist.angular.z),
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # Publish
        self.cmd_pub.publish(twist)
        out_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        out_msg.header = msg.header
        self.img_pub.publish(out_msg)

    # ── Detection ──────────────────────────────────────────────────────
    def _detect_persons(self, frame: np.ndarray) -> List[Tuple[Box, float]]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, scalefactor=0.007843, size=(300, 300),
            mean=(127.5, 127.5, 127.5), swapRB=False, crop=False,
        )
        self.net.setInput(blob)
        output = self.net.forward()

        results = []
        for i in range(output.shape[2]):
            confidence = float(output[0, 0, i, 2])
            class_id = int(output[0, 0, i, 1])
            if class_id != self.PERSON_CLASS_ID:
                continue
            if confidence < self.confidence_threshold:
                continue

            x1 = max(0, min(w - 1, int(output[0, 0, i, 3] * w)))
            y1 = max(0, min(h - 1, int(output[0, 0, i, 4] * h)))
            x2 = max(0, min(w - 1, int(output[0, 0, i, 5] * w)))
            y2 = max(0, min(h - 1, int(output[0, 0, i, 6] * h)))
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            results.append(((x1, y1, bw, bh), confidence))

        return results


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PersonFollowerNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
