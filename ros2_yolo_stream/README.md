# ROS 2 YOLO Stream Demo

This demo subscribes to a ROS camera image, runs lightweight YOLO object
detection with OpenCV DNN, and publishes an annotated image stream.

It avoids PyTorch at runtime. The default model is YOLOv5n exported to ONNX and
converted to FP32 for OpenCV 4.5.x compatibility, which is a practical starting
point for Raspberry Pi-class hardware.

## Build In The Workspace

The Code Server container copies this demo to `/workspace/src/ros2_yolo_stream`
when it starts.

Build it from the workspace root:

```bash
cd /workspace
colcon build --packages-select ros2_yolo_stream
source install/setup.bash
```

## Download The Model

```bash
/workspace/src/ros2_yolo_stream/scripts/download_yolov5n_onnx.sh
```

This stores the model at:

```text
/workspace/models/yolov5n_fp32.onnx
```

## Run

```bash
ros2 launch ros2_yolo_stream yolo_stream.launch.py
```

Default topics:

```text
input:  /image
output: /yolo/image
```

The output can be shown through the video server or RViz as a normal
`sensor_msgs/msg/Image` stream.

## Raspberry Pi Tuning

Use a smaller inference size and skip frames if CPU load is high:

```bash
ros2 launch ros2_yolo_stream yolo_stream.launch.py \
  input_topic:=/image \
  output_topic:=/yolo/image \
  input_size:=320 \
  process_every_n:=3 \
  confidence_threshold:=0.35
```

Lower `input_size` improves speed but reduces accuracy. Increase
`process_every_n` to reduce CPU usage.
