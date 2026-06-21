import os
import threading
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from .control import compute_target_velocity, exponential_smooth

Box = tuple[int, int, int, int]  # x, y, w, h


class PersonFollowerNode(Node):
    """Detect the largest person in the camera image and drive toward them.

    Architecture for smooth control on slow hardware:
      - Detection runs at ~2-3 Hz (limited by DNN inference on CPU).
      - Control loop runs at 20 Hz, interpolating between detections
        with exponential smoothing so the robot moves fluidly.
      - Video output runs at 10 Hz with the latest annotated frame.
    """

    PERSON_CLASS_ID = 15  # VOC class id for "person"

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
        self.declare_parameter("smoothing", 0.15)
        self.declare_parameter("control_hz", 20.0)
        self.declare_parameter("video_hz", 10.0)
        self.declare_parameter("lost_timeout_sec", 2.0)
        self.declare_parameter("search_speed", 0.3)
        self.declare_parameter("enabled", True)

        self.input_topic = self.get_parameter("input_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        self.model_path = self.get_parameter("model_path").value
        self.config_path = self.get_parameter("config_path").value
        self.confidence_threshold = float(self.get_parameter("confidence_threshold").value)
        self.max_linear_speed = float(self.get_parameter("max_linear_speed").value)
        self.max_angular_speed = float(self.get_parameter("max_angular_speed").value)
        self.target_box_ratio = float(self.get_parameter("target_box_ratio").value)
        self.dead_zone = float(self.get_parameter("dead_zone").value)
        # smoothing: 0..1, lower = smoother (more inertia), higher = snappier
        self.smoothing = float(np.clip(self.get_parameter("smoothing").value, 0.02, 1.0))
        self.control_hz = max(5.0, float(self.get_parameter("control_hz").value))
        self.video_hz = max(1.0, float(self.get_parameter("video_hz").value))
        self.lost_timeout = float(self.get_parameter("lost_timeout_sec").value)
        self.search_speed = float(self.get_parameter("search_speed").value)
        self.enabled = bool(self.get_parameter("enabled").value)

        # ── State ──────────────────────────────────────────────────────
        self.bridge = CvBridge()
        self.net = self._load_model()

        self.latest_msg = None
        self.latest_frame_id = 0
        self.last_detect_id = 0
        self._detect_lock = threading.Lock()

        # Shared detection result (written by detect thread, read by control)
        self._target_error_x = 0.0  # -0.5 .. 0.5 (person center offset)
        self._target_size_ratio = 0.0  # bbox width / frame width
        self._target_conf = 0.0
        self._target_box = None  # (x, y, w, h) or None
        self._last_person_time = 0.0
        self._last_detect_frame = None  # latest frame with annotations

        # Smoothed velocity state (updated by control loop)
        self._smooth_vx = 0.0
        self._smooth_wz = 0.0

        # ── Pub / Sub ──────────────────────────────────────────────────
        sub_group = MutuallyExclusiveCallbackGroup()
        detect_group = MutuallyExclusiveCallbackGroup()
        control_group = MutuallyExclusiveCallbackGroup()
        video_group = MutuallyExclusiveCallbackGroup()

        input_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.sub = self.create_subscription(
            Image,
            self.input_topic,
            self._on_image,
            input_qos,
            callback_group=sub_group,
        )
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.img_pub = self.create_publisher(
            Image,
            self.output_topic,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE),
        )

        # Detection timer: as fast as possible (limited by inference)
        self._detect_timer = self.create_timer(
            0.05,
            self._detect_tick,
            callback_group=detect_group,
        )
        # Control timer: 20 Hz - smooth velocity commands
        self._control_timer = self.create_timer(
            1.0 / self.control_hz,
            self._control_tick,
            callback_group=control_group,
        )
        # Video timer: 10 Hz - publish annotated image
        self._video_timer = self.create_timer(
            1.0 / self.video_hz,
            self._video_tick,
            callback_group=video_group,
        )

        self.get_logger().info(
            f"Person follower ready: {self.input_topic} -> {self.cmd_vel_topic}, "
            f"control={int(self.control_hz)}Hz, smooth={self.smoothing:.2f}, "
            f"speed={self.max_linear_speed:.2f}/{self.max_angular_speed:.2f}"
        )

    # ── Model ──────────────────────────────────────────────────────────
    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        net = cv2.dnn.readNetFromCaffe(self.config_path, self.model_path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        # Limit OpenCV internal threads to avoid contention with ROS threads
        cv2.setNumThreads(2)
        return net

    # ── Image callback ─────────────────────────────────────────────────
    def _on_image(self, msg: Image) -> None:
        self.latest_frame_id += 1
        self.latest_msg = msg

    # ── Detection (slow, ~2-3 Hz) ──────────────────────────────────────
    def _detect_tick(self) -> None:
        if not self._detect_lock.acquire(blocking=False):
            return
        try:
            self._run_detection()
        finally:
            self._detect_lock.release()

    def _run_detection(self) -> None:
        msg = self.latest_msg
        if msg is None:
            return
        fid = self.latest_frame_id
        if fid == self.last_detect_id:
            return
        self.last_detect_id = fid

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        h, w = frame.shape[:2]

        # Pre-downscale for faster DNN: resize to ~320px wide before detection.
        # blobFromImage then only has to work on a small image.
        detect_w = 320
        scale_factor = detect_w / w
        detect_h = int(h * scale_factor)
        small = cv2.resize(frame, (detect_w, detect_h), interpolation=cv2.INTER_LINEAR)

        persons = self._detect_persons(small, detect_h, detect_w)
        now = time.monotonic()

        # Map detections back to original frame coordinates
        inv_scale = 1.0 / scale_factor
        persons_orig = []
        for (bx, by, bw, bh), conf in persons:
            persons_orig.append(
                (
                    (
                        int(bx * inv_scale),
                        int(by * inv_scale),
                        int(bw * inv_scale),
                        int(bh * inv_scale),
                    ),
                    conf,
                )
            )

        if persons_orig:
            best = max(persons_orig, key=lambda p: p[0][2] * p[0][3])
            box, conf = best
            bx, by, bw, bh = box
            cx = bx + bw / 2.0

            self._target_error_x = (cx / w) - 0.5
            self._target_size_ratio = bw / w
            self._target_conf = conf
            self._target_box = box
            self._last_person_time = now

            # Annotate
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            label = "FOLLOW %.0f%%" % (conf * 100)
            cv2.putText(
                frame,
                label,
                (bx, max(by - 8, 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        else:
            self._target_box = None
            self._target_conf = 0.0

        # Draw HUD
        has_target = (
            self._target_box is not None and (now - self._last_person_time) < self.lost_timeout
        )
        status = (
            "FOLLOWING"
            if has_target and self.enabled
            else ("SEARCHING" if self.enabled else "DISABLED")
        )
        color = (0, 255, 0) if has_target and self.enabled else (0, 140, 255)
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"v:{self._smooth_vx:.2f} w:{self._smooth_wz:.2f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        self._last_detect_frame = frame
        self._last_detect_header = msg.header

    # ── Control loop (fast, 20 Hz) ─────────────────────────────────────
    def _control_tick(self) -> None:
        if not self.enabled:
            self._smooth_vx = 0.0
            self._smooth_wz = 0.0
            self.cmd_pub.publish(Twist())
            return

        now = time.monotonic()
        person_age = now - self._last_person_time
        has_target = self._target_box is not None and person_age < self.lost_timeout

        # Compute raw desired velocities
        target_vx = 0.0
        target_wz = 0.0

        if has_target:
            target_vx, target_wz = compute_target_velocity(
                self._target_error_x,
                self._target_size_ratio,
                self.target_box_ratio,
                dead_zone=self.dead_zone,
                max_linear_speed=self.max_linear_speed,
                max_angular_speed=self.max_angular_speed,
            )
        elif self._last_person_time > 0 and person_age > self.lost_timeout:
            # Lost person - slowly rotate to search
            target_wz = self.search_speed

        # Exponential smoothing (snaps tiny residual motion to zero)
        a = self.smoothing
        self._smooth_vx = exponential_smooth(self._smooth_vx, target_vx, a)
        self._smooth_wz = exponential_smooth(self._smooth_wz, target_wz, a)

        twist = Twist()
        twist.linear.x = self._smooth_vx
        twist.angular.z = self._smooth_wz
        self.cmd_pub.publish(twist)

    # ── Video output (10 Hz) ───────────────────────────────────────────
    def _video_tick(self) -> None:
        frame = self._last_detect_frame
        if frame is None:
            return
        out_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        out_msg.header = self._last_detect_header
        self.img_pub.publish(out_msg)

    # ── Detection ──────────────────────────────────────────────────────
    def _detect_persons(self, frame: np.ndarray, h: int, w: int) -> list[tuple[Box, float]]:
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=0.007843,
            size=(300, 300),
            mean=(127.5, 127.5, 127.5),
            swapRB=False,
            crop=False,
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
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
