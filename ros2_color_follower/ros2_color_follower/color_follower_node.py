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
    """Follow a colored ball using hybrid HSV detection + CamShift tracking.

    Strategy:
      1. HSV color filter + circularity to FIND the ball initially.
      2. Once found, build a Hue histogram of the ball region.
      3. Use CamShift (histogram back-projection) to TRACK frame-to-frame.
         CamShift adapts to lighting changes automatically.
      4. Periodically re-verify with HSV detection; if CamShift drifts,
         fall back to full HSV re-detection.
      5. When lost, just stop – no annoying search rotation.
    """

    def __init__(self) -> None:
        super().__init__("color_follower")

        # ── Parameters ─────────────────────────────────────────────────
        self.declare_parameter("input_topic", "/image")
        self.declare_parameter("output_topic", "/follower/image")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("hsv_low_h", 25)
        self.declare_parameter("hsv_low_s", 40)
        self.declare_parameter("hsv_low_v", 40)
        self.declare_parameter("hsv_high_h", 95)
        self.declare_parameter("hsv_high_s", 255)
        self.declare_parameter("hsv_high_v", 255)
        self.declare_parameter("min_area", 200)
        self.declare_parameter("min_circularity", 0.40)
        self.declare_parameter("max_linear_speed", 0.3)
        self.declare_parameter("max_angular_speed", 1.0)
        self.declare_parameter("target_radius_ratio", 0.12)
        self.declare_parameter("dead_zone", 0.04)
        self.declare_parameter("smoothing", 0.25)
        self.declare_parameter("control_hz", 20.0)
        self.declare_parameter("video_hz", 15.0)
        self.declare_parameter("lost_timeout_sec", 1.5)
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
        self.enabled = bool(self.get_parameter("enabled").value)

        # ── Tracker state ──────────────────────────────────────────────
        self.bridge = CvBridge()
        self.latest_msg = None
        self.latest_frame_id = 0
        self.last_detect_id = 0
        self._detect_lock = threading.Lock()

        # CamShift tracking state
        self._track_window = None   # (x, y, w, h) in downscaled coords
        self._track_hist = None     # Hue histogram for back-projection
        self._track_scale = 1.0     # downscale factor
        self._camshift_fails = 0    # consecutive CamShift failures
        self._verify_counter = 0    # frames since last HSV re-verification
        self._VERIFY_INTERVAL = 15  # re-verify with HSV every N frames
        self._MAX_CAMSHIFT_FAILS = 5

        # Shared output for control loop
        self._target_error_x = 0.0
        self._target_radius_ratio = 0.0
        self._target_found = False
        self._last_seen_time = 0.0

        # Smoothed velocities
        self._smooth_vx = 0.0
        self._smooth_wz = 0.0

        # Latest annotated frame
        self._last_frame = None
        self._last_header = None

        # CamShift termination criteria
        self._term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

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

        self._detect_timer = self.create_timer(
            0.03, self._detect_tick, callback_group=detect_group,
        )
        self._control_timer = self.create_timer(
            1.0 / self.control_hz, self._control_tick, callback_group=control_group,
        )
        self._video_timer = self.create_timer(
            1.0 / self.video_hz, self._video_tick, callback_group=video_group,
        )

        self.get_logger().info(
            "Color follower ready (hybrid HSV+CamShift): HSV [%d,%d,%d]-[%d,%d,%d]"
            % (*self.hsv_low, *self.hsv_high)
        )

    # ── Image callback ─────────────────────────────────────────────────
    def _on_image(self, msg: Image) -> None:
        self.latest_frame_id += 1
        self.latest_msg = msg

    # ── Main detection / tracking loop ─────────────────────────────────
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

        # Downscale for processing
        proc_w = 320
        scale = proc_w / w
        proc_h = int(h * scale)
        self._track_scale = scale
        small = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)
        blurred = cv2.GaussianBlur(small, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        now = time.monotonic()
        found = False
        cx_full = 0.0
        r_full = 0.0
        method = ""

        # ── Strategy: CamShift track if we have a histogram ────────────
        if self._track_hist is not None and self._track_window is not None:
            self._verify_counter += 1

            # Back-project the stored hue histogram
            mask_s = cv2.inRange(hsv, self.hsv_low, self.hsv_high)
            hue = hsv[:, :, 0]
            back_proj = cv2.calcBackProject([hue], [0], self._track_hist, [0, 180], 1)
            back_proj &= mask_s  # combine with loose HSV mask

            try:
                ret, window = cv2.CamShift(back_proj, self._track_window, self._term_crit)
                (cx_s, cy_s), (bw_s, bh_s), angle = ret
                tx, ty, tw, th = window

                if tw > 5 and th > 5 and bw_s > 5 and bh_s > 5:
                    self._track_window = window
                    self._camshift_fails = 0

                    cx_full = cx_s / scale
                    r_full = max(bw_s, bh_s) / (2.0 * scale)
                    found = True
                    method = "CAMSHIFT"
                else:
                    self._camshift_fails += 1
            except cv2.error:
                self._camshift_fails += 1

            # Periodically re-verify with full HSV detection to prevent drift
            need_verify = self._verify_counter >= self._VERIFY_INTERVAL
            if self._camshift_fails >= self._MAX_CAMSHIFT_FAILS or need_verify:
                hsv_result = self._hsv_detect(blurred, hsv, proc_w, proc_h)
                self._verify_counter = 0

                if hsv_result is not None:
                    scx, scy, sr, cnt = hsv_result
                    # Update tracking window and refresh histogram
                    self._init_tracker(hsv, scx, scy, sr, cnt, proc_w, proc_h)
                    cx_full = scx / scale
                    r_full = sr / scale
                    found = True
                    method = "RE-LOCK"
                    self._camshift_fails = 0
                elif self._camshift_fails >= self._MAX_CAMSHIFT_FAILS:
                    # Lost it – reset tracker
                    self._track_hist = None
                    self._track_window = None
                    found = False
        else:
            # ── No active tracker – full HSV detection to acquire ──────
            hsv_result = self._hsv_detect(blurred, hsv, proc_w, proc_h)
            if hsv_result is not None:
                scx, scy, sr, cnt = hsv_result
                self._init_tracker(hsv, scx, scy, sr, cnt, proc_w, proc_h)
                cx_full = scx / scale
                r_full = sr / scale
                found = True
                method = "DETECT"

        # ── Update shared state ────────────────────────────────────────
        if found:
            self._target_error_x = (cx_full / w) - 0.5
            self._target_radius_ratio = (r_full * 2.0) / w
            self._target_found = True
            self._last_seen_time = now

            center = (int(cx_full), int(h / 2) if r_full == 0 else int(cx_full * h / w))
            # Recalculate center y properly
            center = (int(cx_full), int(frame.shape[0] / 2))
            if self._track_window is not None:
                tx, ty, tw, th = self._track_window
                cy_full = (ty + th / 2.0) / scale
                center = (int(cx_full), int(cy_full))
            radius_px = max(10, int(r_full))

            cv2.circle(frame, center, radius_px, (0, 255, 0), 2)
            cv2.circle(frame, center, 4, (0, 255, 0), -1)
            cv2.putText(frame, method, (center[0] + radius_px + 5, center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        else:
            self._target_found = False

        # HUD
        age = now - self._last_seen_time if self._last_seen_time > 0 else 999
        tracking = self._target_found and age < self.lost_timeout
        status = "TRACKING" if tracking and self.enabled else (
            "LOST" if self.enabled else "DISABLED"
        )
        hud_color = (0, 255, 0) if tracking and self.enabled else (0, 100, 255)
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, hud_color, 2, cv2.LINE_AA)
        cv2.putText(frame, "v:%.2f w:%.2f" % (self._smooth_vx, self._smooth_wz),
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        self._last_frame = frame
        self._last_header = msg.header

    # ── HSV color + circularity detection ──────────────────────────────
    def _hsv_detect(self, blurred, hsv, pw, ph):
        """Return (cx, cy, radius, contour) in downscaled coords, or None."""
        mask = cv2.inRange(hsv, self.hsv_low, self.hsv_high)
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=3)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
            score = area * circularity
            if score > best_score:
                best_score = score
                best_match = cnt

        if best_match is None:
            return None

        ((cx, cy), radius) = cv2.minEnclosingCircle(best_match)
        return (cx, cy, radius, best_match)

    # ── Initialize CamShift tracker from a detection ───────────────────
    def _init_tracker(self, hsv, cx, cy, radius, cnt, pw, ph):
        """Build a hue histogram from the ball region for CamShift."""
        r = max(int(radius), 10)
        x1 = max(0, int(cx - r))
        y1 = max(0, int(cy - r))
        x2 = min(pw, int(cx + r))
        y2 = min(ph, int(cy + r))
        if x2 - x1 < 5 or y2 - y1 < 5:
            return

        self._track_window = (x1, y1, x2 - x1, y2 - y1)

        # Build hue histogram from the ball ROI, masked to the HSV range
        roi_hsv = hsv[y1:y2, x1:x2]
        roi_mask = cv2.inRange(roi_hsv, self.hsv_low, self.hsv_high)
        hist = cv2.calcHist([roi_hsv], [0], roi_mask, [32], [0, 180])
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        self._track_hist = hist
        self._camshift_fails = 0
        self._verify_counter = 0

    # ── Control loop (20 Hz) ───────────────────────────────────────────
    def _control_tick(self) -> None:
        if not self.enabled:
            self._smooth_vx = 0.0
            self._smooth_wz = 0.0
            self.cmd_pub.publish(Twist())
            return

        now = time.monotonic()
        age = now - self._last_seen_time if self._last_seen_time > 0 else 999
        has_target = self._target_found and age < self.lost_timeout

        target_vx = 0.0
        target_wz = 0.0

        if has_target:
            ex = self._target_error_x
            if abs(ex) > self.dead_zone:
                target_wz = -ex * 2.0 * self.max_angular_speed

            size_error = self.target_radius_ratio - self._target_radius_ratio
            if abs(size_error) > 0.02:
                # Quadratic response: faster when ball is far, gentle when close
                sign = 1.0 if size_error > 0 else -1.0
                target_vx = float(np.clip(
                    sign * (size_error ** 2) * 30.0 * self.max_linear_speed + size_error * 1.5 * self.max_linear_speed,
                    -self.max_linear_speed,
                    self.max_linear_speed,
                ))
        # No search rotation – just stop when lost

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
