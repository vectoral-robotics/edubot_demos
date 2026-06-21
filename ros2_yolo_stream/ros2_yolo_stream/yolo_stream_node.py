import os
import threading
import time
from collections.abc import Sequence

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

Box = tuple[int, int, int, int]


class YoloStreamNode(Node):
    def __init__(self) -> None:
        super().__init__("yolo_stream")

        self.declare_parameter("input_topic", "/image")
        self.declare_parameter("output_topic", "/yolo/image")
        self.declare_parameter("model_type", "ssd")
        self.declare_parameter("model_path", "/workspace/models/mobilenet_iter_73000.caffemodel")
        self.declare_parameter("config_path", "/workspace/models/mobilenet_ssd.prototxt")
        self.declare_parameter("class_names_path", self._default_class_names_path())
        self.declare_parameter("input_size", 320)
        self.declare_parameter("confidence_threshold", 0.35)
        self.declare_parameter("nms_threshold", 0.45)
        self.declare_parameter("process_every_n", 1)
        self.declare_parameter("max_processing_fps", 2.0)
        self.declare_parameter("output_fps", 5.0)
        self.declare_parameter("output_reliability", "reliable")
        self.declare_parameter("watchdog_timeout_sec", 5.0)

        self.input_topic = self.get_parameter("input_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.model_type = str(self.get_parameter("model_type").value).lower()
        self.model_path = self.get_parameter("model_path").value
        self.config_path = self.get_parameter("config_path").value
        self.class_names_path = self.get_parameter("class_names_path").value
        self.input_size = int(self.get_parameter("input_size").value)
        self.confidence_threshold = float(self.get_parameter("confidence_threshold").value)
        self.nms_threshold = float(self.get_parameter("nms_threshold").value)
        self.process_every_n = max(1, int(self.get_parameter("process_every_n").value))
        self.max_processing_fps = max(0.1, float(self.get_parameter("max_processing_fps").value))
        self.output_fps = max(0.1, float(self.get_parameter("output_fps").value))
        self.output_reliability = str(self.get_parameter("output_reliability").value).lower()
        self.watchdog_timeout_sec = max(
            1.0, float(self.get_parameter("watchdog_timeout_sec").value)
        )

        self.bridge = CvBridge()
        self.class_names = self._load_class_names(self.class_names_path)
        self.net = self._load_model(self.model_path)
        self.frame_count = 0
        self.published_count = 0
        self.processed_count = 0
        self.latest_msg = None
        self.latest_frame_count = 0
        self.last_published_frame_count = 0
        self.last_processed_frame_count = 0
        self.last_detections: list[tuple[Box, float, int]] = []
        self.last_input_time = time.monotonic()
        self.last_publish_time = time.monotonic()
        self.last_log_time = time.monotonic()
        self.last_watchdog_log_time = 0.0
        self.inference_error_logged = False
        self._processing_lock = threading.Lock()

        # Separate callback groups so the subscription never gets starved
        # by a long-running inference timer callback.
        sub_cb_group = MutuallyExclusiveCallbackGroup()
        timer_cb_group = MutuallyExclusiveCallbackGroup()

        reliability = (
            ReliabilityPolicy.BEST_EFFORT
            if self.output_reliability == "best_effort"
            else ReliabilityPolicy.RELIABLE
        )
        output_qos = QoSProfile(depth=1, reliability=reliability)
        self.publisher = self.create_publisher(Image, self.output_topic, output_qos)
        # Use RELIABLE to match cam2image's default QoS; keep only the latest
        # frame to avoid buffering lag.
        input_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.subscription = self.create_subscription(
            Image,
            self.input_topic,
            self._on_image,
            input_qos,
            callback_group=sub_cb_group,
        )
        self.processing_timer = self.create_timer(
            1.0 / self.max_processing_fps,
            self._process_latest_image,
            callback_group=timer_cb_group,
        )
        self.output_timer = self.create_timer(
            1.0 / self.output_fps,
            self._publish_latest_image,
            callback_group=timer_cb_group,
        )
        self.watchdog_timer = self.create_timer(1.0, self._watchdog, callback_group=timer_cb_group)

        self.get_logger().info(
            f"Detection stream ready: {self.input_topic} -> {self.output_topic}, type={self.model_type}, model={self.model_path}, output_fps={self.output_fps:.1f}, max_processing_fps={self.max_processing_fps:.1f}, output_reliability={self.output_reliability}"
        )

    def _default_class_names_path(self) -> str:
        share_dir = get_package_share_directory("ros2_yolo_stream")
        return os.path.join(share_dir, "config", "voc.names")

    def _load_class_names(self, path: str) -> list[str]:
        with open(path, encoding="utf-8") as names_file:
            return [line.strip() for line in names_file if line.strip()]

    def _load_model(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Detection model not found at {path}. Run the matching download script first."
            )

        if self.model_type == "ssd":
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(
                    f"SSD config not found at {self.config_path}. Run scripts/download_mobilenet_ssd.sh first."
                )
            net = cv2.dnn.readNetFromCaffe(self.config_path, path)
        else:
            net = cv2.dnn.readNetFromONNX(path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return net

    def _on_image(self, msg: Image) -> None:
        self.frame_count += 1
        self.latest_msg = msg
        self.latest_frame_count = self.frame_count
        self.last_input_time = time.monotonic()

    def _publish_latest_image(self) -> None:
        msg = self.latest_msg
        if msg is None:
            return
        if self.latest_frame_count == self.last_published_frame_count:
            return

        self.last_published_frame_count = self.latest_frame_count
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        annotated = self._draw_detections(frame, self.last_detections)
        out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        out_msg.header = msg.header
        self.publisher.publish(out_msg)
        self.published_count += 1
        self.last_publish_time = time.monotonic()

    def _process_latest_image(self) -> None:
        if not self._processing_lock.acquire(blocking=False):
            return  # previous inference still running
        try:
            self._process_latest_image_locked()
        finally:
            self._processing_lock.release()

    def _process_latest_image_locked(self) -> None:
        msg = self.latest_msg
        if msg is None:
            return
        if self.latest_frame_count == self.last_processed_frame_count:
            return
        if self.latest_frame_count - self.last_processed_frame_count < self.process_every_n:
            return

        self.last_processed_frame_count = self.latest_frame_count
        start = time.monotonic()

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            detections = self._detect(frame.copy())
            self.last_detections = detections
        except cv2.error as exc:
            detections = []
            if not self.inference_error_logged:
                self.get_logger().error(f"OpenCV DNN inference failed: {exc}")
                self.inference_error_logged = True

        self.processed_count += 1
        now = time.monotonic()
        if now - self.last_log_time > 10.0:
            elapsed_ms = (now - start) * 1000.0
            self.get_logger().info(
                f"Received {self.frame_count} frames, published {self.published_count}, "
                f"processed {self.processed_count}, last inference {elapsed_ms:.1f} ms, "
                f"detections={len(detections)}"
            )
            self.last_log_time = now

    def _watchdog(self) -> None:
        now = time.monotonic()
        no_input_for = now - self.last_input_time
        no_publish_for = now - self.last_publish_time
        if no_input_for < self.watchdog_timeout_sec and no_publish_for < self.watchdog_timeout_sec:
            return
        if now - self.last_watchdog_log_time < self.watchdog_timeout_sec:
            return

        self.get_logger().warn(
            f"Watchdog: no input for {no_input_for:.1f}s, no output for {no_publish_for:.1f}s, "
            f"received={self.frame_count}, published={self.published_count}, "
            f"processed={self.processed_count}"
        )
        self.last_watchdog_log_time = now

    def _detect(self, frame: np.ndarray) -> list[tuple[Box, float, int]]:
        if self.model_type == "ssd":
            return self._detect_ssd(frame)
        return self._detect_yolo(frame)

    def _detect_ssd(self, frame: np.ndarray) -> list[tuple[Box, float, int]]:
        height, width = frame.shape[:2]
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

        detections: list[tuple[Box, float, int]] = []
        for index in range(output.shape[2]):
            confidence = float(output[0, 0, index, 2])
            if confidence < self.confidence_threshold:
                continue

            class_id = int(output[0, 0, index, 1])
            x1 = max(0, min(width - 1, int(output[0, 0, index, 3] * width)))
            y1 = max(0, min(height - 1, int(output[0, 0, index, 4] * height)))
            x2 = max(0, min(width - 1, int(output[0, 0, index, 5] * width)))
            y2 = max(0, min(height - 1, int(output[0, 0, index, 6] * height)))
            detections.append(((x1, y1, max(1, x2 - x1), max(1, y2 - y1)), confidence, class_id))

        return detections

    def _detect_yolo(self, frame: np.ndarray) -> list[tuple[Box, float, int]]:
        input_image, scale, pad_x, pad_y = self._letterbox(frame, self.input_size)
        blob = cv2.dnn.blobFromImage(
            input_image,
            scalefactor=1.0 / 255.0,
            size=(self.input_size, self.input_size),
            mean=(0, 0, 0),
            swapRB=True,
            crop=False,
        )

        self.net.setInput(blob)
        output = self.net.forward()
        rows = self._normalize_output(output)

        boxes: list[Box] = []
        confidences: list[float] = []
        class_ids: list[int] = []
        height, width = frame.shape[:2]

        for row in rows:
            objectness = float(row[4])
            if objectness < self.confidence_threshold:
                continue

            class_scores = row[5:]
            class_id = int(np.argmax(class_scores))
            confidence = objectness * float(class_scores[class_id])
            if confidence < self.confidence_threshold:
                continue

            center_x, center_y, box_width, box_height = row[:4]
            x = (float(center_x) - float(box_width) / 2.0 - pad_x) / scale
            y = (float(center_y) - float(box_height) / 2.0 - pad_y) / scale
            w = float(box_width) / scale
            h = float(box_height) / scale

            x1 = max(0, min(width - 1, round(x)))
            y1 = max(0, min(height - 1, round(y)))
            x2 = max(0, min(width - 1, round(x + w)))
            y2 = max(0, min(height - 1, round(y + h)))
            clipped_w = max(1, x2 - x1)
            clipped_h = max(1, y2 - y1)

            boxes.append((x1, y1, clipped_w, clipped_h))
            confidences.append(confidence)
            class_ids.append(class_id)

        selected = cv2.dnn.NMSBoxes(
            boxes,
            confidences,
            self.confidence_threshold,
            self.nms_threshold,
        )

        detections: list[tuple[Box, float, int]] = []
        for index in self._flatten_indices(selected):
            detections.append((boxes[index], confidences[index], class_ids[index]))
        return detections

    def _normalize_output(self, output: np.ndarray) -> np.ndarray:
        rows = np.squeeze(output)
        if rows.ndim == 1:
            rows = np.expand_dims(rows, axis=0)
        if rows.shape[0] < rows.shape[-1] and rows.shape[0] < 100:
            rows = rows.T
        return rows

    def _letterbox(
        self, frame: np.ndarray, target_size: int
    ) -> tuple[np.ndarray, float, int, int]:
        height, width = frame.shape[:2]
        scale = min(target_size / width, target_size / height)
        resized_width = round(width * scale)
        resized_height = round(height * scale)
        pad_x = (target_size - resized_width) // 2
        pad_y = (target_size - resized_height) // 2

        resized = cv2.resize(
            frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
        )
        canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
        return canvas, scale, pad_x, pad_y

    def _draw_detections(
        self,
        frame: np.ndarray,
        detections: Sequence[tuple[Box, float, int]],
    ) -> np.ndarray:
        annotated = frame.copy()
        for box, confidence, class_id in detections:
            x, y, w, h = box
            label = (
                self.class_names[class_id] if class_id < len(self.class_names) else str(class_id)
            )
            text = f"{label} {confidence:.2f}"

            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 220, 0), 2)
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_w, text_h = text_size
            text_y = max(y, text_h + 6)
            cv2.rectangle(
                annotated,
                (x, text_y - text_h - 6),
                (x + text_w + 6, text_y + 2),
                (0, 220, 0),
                -1,
            )
            cv2.putText(
                annotated,
                text,
                (x + 3, text_y - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
        return annotated

    def _draw_error(self, frame: np.ndarray, text: str) -> np.ndarray:
        annotated = frame.copy()
        cv2.rectangle(annotated, (8, 8), (420, 44), (0, 0, 180), -1)
        cv2.putText(
            annotated,
            text,
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return annotated

    def _flatten_indices(self, indices) -> list[int]:
        if indices is None or len(indices) == 0:
            return []
        return [int(index) for index in np.array(indices).reshape(-1)]


def main(args=None) -> None:
    rclpy.init(args=args)
    node = YoloStreamNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
