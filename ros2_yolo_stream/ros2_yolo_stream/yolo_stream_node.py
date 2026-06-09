import os
import time
from typing import List, Sequence, Tuple

import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


Box = Tuple[int, int, int, int]


class YoloStreamNode(Node):
    def __init__(self) -> None:
        super().__init__("yolo_stream")

        self.declare_parameter("input_topic", "/image")
        self.declare_parameter("output_topic", "/yolo/image")
        self.declare_parameter("model_path", "/workspace/models/yolov5n.onnx")
        self.declare_parameter("class_names_path", self._default_class_names_path())
        self.declare_parameter("input_size", 320)
        self.declare_parameter("confidence_threshold", 0.35)
        self.declare_parameter("nms_threshold", 0.45)
        self.declare_parameter("process_every_n", 3)

        self.input_topic = self.get_parameter("input_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.model_path = self.get_parameter("model_path").value
        self.class_names_path = self.get_parameter("class_names_path").value
        self.input_size = int(self.get_parameter("input_size").value)
        self.confidence_threshold = float(self.get_parameter("confidence_threshold").value)
        self.nms_threshold = float(self.get_parameter("nms_threshold").value)
        self.process_every_n = max(1, int(self.get_parameter("process_every_n").value))

        self.bridge = CvBridge()
        self.class_names = self._load_class_names(self.class_names_path)
        self.net = self._load_model(self.model_path)
        self.frame_count = 0
        self.processed_count = 0
        self.last_log_time = time.monotonic()
        self.inference_error_logged = False

        self.publisher = self.create_publisher(Image, self.output_topic, qos_profile_sensor_data)
        self.subscription = self.create_subscription(
            Image,
            self.input_topic,
            self._on_image,
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            "YOLO stream ready: %s -> %s, model=%s, input_size=%d, process_every_n=%d"
            % (
                self.input_topic,
                self.output_topic,
                self.model_path,
                self.input_size,
                self.process_every_n,
            )
        )

    def _default_class_names_path(self) -> str:
        share_dir = get_package_share_directory("ros2_yolo_stream")
        return os.path.join(share_dir, "config", "coco.names")

    def _load_class_names(self, path: str) -> List[str]:
        with open(path, "r", encoding="utf-8") as names_file:
            return [line.strip() for line in names_file if line.strip()]

    def _load_model(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(
                "YOLO model not found at %s. Run scripts/download_yolov5n_onnx.sh first." % path
            )

        net = cv2.dnn.readNetFromONNX(path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return net

    def _on_image(self, msg: Image) -> None:
        self.frame_count += 1
        if self.frame_count % self.process_every_n != 0:
            return

        start = time.monotonic()

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        try:
            detections = self._detect(frame)
            annotated = self._draw_detections(frame, detections)
        except cv2.error as exc:
            detections = []
            annotated = self._draw_error(frame, "DNN error; try input_size:=640")
            if not self.inference_error_logged:
                self.get_logger().error("OpenCV DNN inference failed: %s" % exc)
                self.inference_error_logged = True

        out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        out_msg.header = msg.header
        self.publisher.publish(out_msg)

        self.processed_count += 1
        now = time.monotonic()
        if now - self.last_log_time > 10.0:
            elapsed_ms = (now - start) * 1000.0
            self.get_logger().info(
                "Processed %d frames, last inference+publish %.1f ms, detections=%d"
                % (self.processed_count, elapsed_ms, len(detections))
            )
            self.last_log_time = now

    def _detect(self, frame: np.ndarray) -> List[Tuple[Box, float, int]]:
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

        boxes: List[Box] = []
        confidences: List[float] = []
        class_ids: List[int] = []
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

            x1 = max(0, min(width - 1, int(round(x))))
            y1 = max(0, min(height - 1, int(round(y))))
            x2 = max(0, min(width - 1, int(round(x + w))))
            y2 = max(0, min(height - 1, int(round(y + h))))
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

        detections: List[Tuple[Box, float, int]] = []
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

    def _letterbox(self, frame: np.ndarray, target_size: int) -> Tuple[np.ndarray, float, int, int]:
        height, width = frame.shape[:2]
        scale = min(target_size / width, target_size / height)
        resized_width = int(round(width * scale))
        resized_height = int(round(height * scale))
        pad_x = (target_size - resized_width) // 2
        pad_y = (target_size - resized_height) // 2

        resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
        return canvas, scale, pad_x, pad_y

    def _draw_detections(
        self,
        frame: np.ndarray,
        detections: Sequence[Tuple[Box, float, int]],
    ) -> np.ndarray:
        annotated = frame.copy()
        for box, confidence, class_id in detections:
            x, y, w, h = box
            label = self.class_names[class_id] if class_id < len(self.class_names) else str(class_id)
            text = "%s %.2f" % (label, confidence)

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

    def _flatten_indices(self, indices) -> List[int]:
        if indices is None or len(indices) == 0:
            return []
        return [int(index) for index in np.array(indices).reshape(-1)]


def main(args=None) -> None:
    rclpy.init(args=args)
    node = YoloStreamNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
