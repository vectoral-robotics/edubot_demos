import time
import threading

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


class ColorFollowerNode(Node):
    """Follow a colored object (e.g. bright-green ball) using HSV filtering.

    Runs at full camera frame rate with almost no CPU load.
    A 20 Hz control loop with exponential smoothing keeps motion fluid.
    """

    def __init__(self) -> None:
        super().__init__("color_follower")

        # ── Parameters ─────────────────────────────────────────────────
        self.declare_parameter("input_topic", "/image")
        self.declare_parameter("output_topic", "/follower/image")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        # HSV range for bright / lime green  (tune with a color picker)
        self.declare_parameter("hsv_low_h", 25)
        self.declare_parameter("hsv_low_s", 40)
        self.declare_parameter("hsv_low_v", 40)
        self.declare_parameter("hsv_high_h", 95)
        self.declare_parameter("hsv_high_s", 255)
        self.declare_parameter("hsv_high_v", 255)
        self.declare_parameter("min_area", 200)
        self.declare_parameter("min_circularity", 0.45)
        self.declare_parameter("max_linear_speed", 0.3)
        self.declare_parameter("max_angular_speed", 1.0)
        self.declare_parameter("target_radius_ratio", 0.12)
        self.declare_parameter("dead_zone", 0.04)
        self.declare_parameter("smoothing", 0.20)
        self.declare_parameter("control_hz", 20.0)
        self.declare_parameter("video_hz", 15.0)
        self.declare_parameter("lost_timeout_sec", 1.5)
        self.declare_parameter("search_speed", 0.4)
        self.declare_parameter("enabled", True)

        self.input_topic = self.get_parameter("input_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        self.hsv_low = np.array([
            self.get_parameter("hsv_low_h").value,
            self.get_parameter("hsv_low_s").value,
            self.get_parameter("hsv_low_v").value,
        ], dtype=np.uint8)
        self.hsv_high = np.array([
            self.get_parameter("hsv_high_h").value,
            self.get_parameter("hsv_high_s").value,
            self.get_parameter("hsv_high_v").value,
        ], dtype=np.uint8)
        self.min_area = int(self.get_parameter("min_area").value)
        self.min_circularity = float(self.get_parameter("min_circularity").value)
        self.max_linear_speed = float(self.get_parameter("max_linear_speed").value)
        self.max_angular_speed = float(self.get_parameter("max_angular_speed").value)
        self.target_radius_ratio = float(self.get_parameter("target_radius_ratio").value)
        self.dead_zone = float(self.get_parameter("dead_zone").value)
        self.smoothing = float(np.clip(self.get_parameter("smoothing").value, 0.02, 1.0))
        self.control_hz = max(5.0, float(self.get_parameter("control_hz").value))
        self.video_hz = max(1.0, float(self.get_parameter("video_hz").value))
        self.lost_timeout = float(self.get_parameter("lost_timeout_sec").value)
        self.search_speed = float(self.get_parameter("search_speed").value)
        self.enabled = bool(self.get_parameter("enabled").value)

        # ── State ──────────────────────────────────────────────────────
        self.bridge = CvBridge()
        self.latest_msg = None
        self.latest_frame_id = 0
        self.last_detect_id = 0
        self._detect_lock = threading.Lock()

        # Detection result (updated every frame)
        self._target_error_x = 0.0
        self._target_radius_ratio = 0.0
        self._target_center = None
        self._target_radius = 0
        self._last_seen_time = 0.0

        # Smoothed velocities
        self._smooth_vx = 0.0
        self._smooth_wz = 0.0

        # Latest annotated frame for video output
        self._last_frame = None
        self._last_header = None

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
            Image, self.input_topic, self._on_image, input_qos,
            callback_group=sub_group,
        )
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.img_pub = self.create_publisher(
            Image, self.output_topic,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE),
        )

        # Detection: as fast as frames come in
        self._detect_timer = self.create_timer(
            0.03, self._detect_tick, callback_group=detect_group,
        )
        # Control: 20 Hz smooth velocity
        self._control_timer = self.create_timer(
            1.0 / self.control_hz, self._control_tick, callback_group=control_group,
        )
        # Video: 15 Hz annotated output
        self._video_timer = self.create_timer(
            1.0 / self.video_hz, self._video_tick, callback_group=video_group,
        )

        self.get_logger().info(
            "Color follower ready: HSV [%d,%d,%d]-[%d,%d,%d], control=%dHz, speed=%.2f/%.2f"
            % (*self.hsv_low, *self.hsv_high,
               int(self.control_hz), self.max_linear_speed, self.max_angular_speed)
        )

    # ── Image callback ─────────────────────────────────────────────────
    def _on_image(self, msg: Image) -> None:
        self.latest_frame_id += 1
        self.latest_msg = msg

    # ── Detection (fast, every frame) ──────────────────────────────────
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

        # Downscale for processing (keeps drawing on full frame)
        proc_w = 320
        scale = proc_w / w
        proc_h = int(h * scale)
        small = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)

        # Blur to handle noise and color gradients
        blurred = cv2.GaussianBlur(small, (11, 11), 0)

        # HSV filter
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_low, self.hsv_high)

        # Gentle morphology: small erode to kill speckles, larger dilate to
        # connect fragmented blobs (e.g. specular highlights on the ball)
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=3)

        # Find contours and pick the best ball-shaped one (color + shape)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        now = time.monotonic()
        found = False

        # Score each contour by area, only accept round shapes
        best_score = 0.0
        best_match = None

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter < 1.0:
                continue
            circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
            if circularity < self.min_circularity:
                continue
            # Score: prefer large + round blobs
            score = area * circularity
            if score > best_score:
                best_score = score
                best_match = cnt

        if best_match is not None:
            area = cv2.contourArea(best_match)
            ((cx, cy), radius) = cv2.minEnclosingCircle(best_match)
                # Scale back to full frame
                cx_full = cx / scale
                cy_full = cy / scale
                r_full = radius / scale

                self._target_error_x = (cx_full / w) - 0.5
                self._target_radius_ratio = (r_full * 2.0) / w
                self._target_center = (int(cx_full), int(cy_full))
                self._target_radius = int(r_full)
                self._last_seen_time = now
                found = True

                # Draw on full frame
                cv2.circle(frame, self._target_center, self._target_radius, (0, 255, 0), 2)
                cv2.circle(frame, self._target_center, 4, (0, 255, 0), -1)

        if not found:
            self._target_center = None

        # HUD
        has_target = self._target_center is not None and (now - self._last_seen_time) < self.lost_timeout
        status = "TRACKING" if has_target and self.enabled else (
            "SEARCHING" if self.enabled else "DISABLED"
        )
        color = (0, 255, 0) if has_target and self.enabled else (0, 140, 255)
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        cv2.putText(frame, "v:%.2f w:%.2f" % (self._smooth_vx, self._smooth_wz),
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        self._last_frame = frame
        self._last_header = msg.header

    # ── Control loop (20 Hz) ───────────────────────────────────────────
    def _control_tick(self) -> None:
        if not self.enabled:
            self._smooth_vx = 0.0
            self._smooth_wz = 0.0
            self.cmd_pub.publish(Twist())
            return

        now = time.monotonic()
        age = now - self._last_seen_time
        has_target = self._target_center is not None and age < self.lost_timeout

        target_vx = 0.0
        target_wz = 0.0

        if has_target:
            # Angular: steer toward blob center
            ex = self._target_error_x
            if abs(ex) > self.dead_zone:
                target_wz = -ex * 2.0 * self.max_angular_speed

            # Linear: drive based on apparent size
            size_error = self.target_radius_ratio - self._target_radius_ratio
            if abs(size_error) > 0.02:
                target_vx = float(np.clip(
                    size_error * 3.0 * self.max_linear_speed,
                    -self.max_linear_speed,
                    self.max_linear_speed,
                ))
        elif self._last_seen_time > 0 and age > self.lost_timeout:
            target_wz = self.search_speed

        # Exponential smoothing
        a = self.smoothing
        self._smooth_vx = a * target_vx + (1.0 - a) * self._smooth_vx
        self._smooth_wz = a * target_wz + (1.0 - a) * self._smooth_wz

        if abs(self._smooth_vx) < 0.01:
            self._smooth_vx = 0.0
        if abs(self._smooth_wz) < 0.01:
            self._smooth_wz = 0.0

        twist = Twist()
        twist.linear.x = self._smooth_vx
        twist.angular.z = self._smooth_wz
        self.cmd_pub.publish(twist)

    # ── Video output ───────────────────────────────────────────────────
    def _video_tick(self) -> None:
        frame = self._last_frame
        if frame is None:
            return
        out_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        out_msg.header = self._last_header
        self.img_pub.publish(out_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ColorFollowerNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
