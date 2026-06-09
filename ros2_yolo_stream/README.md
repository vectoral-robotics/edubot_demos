# ROS 2 YOLO Stream Demo

This demo subscribes to a ROS camera image, runs lightweight YOLO object
detection with OpenCV DNN, and publishes an annotated image stream.

It avoids PyTorch at runtime. The default model is YOLOv5n exported to ONNX,
which is a practical starting point for Raspberry Pi-class hardware.

## Install Into The Workspace

The Code Server container copies this demo to `/workspace/demos/ros2_yolo_stream`
when it starts.

For ROS 2 builds, copy it into the workspace source folder:

```bash
mkdir -p /workspace/src
cp -a /workspace/demos/ros2_yolo_stream /workspace/src/
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
/workspace/models/yolov5n.onnx
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
