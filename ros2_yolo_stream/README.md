# ROS 2 YOLO Stream Demo

This demo subscribes to a ROS camera image, runs lightweight object detection
with OpenCV DNN, and publishes an annotated image stream.

It avoids PyTorch at runtime. The default model is MobileNet SSD because it is
small and works with the OpenCV 4.5.x version shipped with ROS Humble images.

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
/workspace/src/ros2_yolo_stream/scripts/download_mobilenet_ssd.sh
```

This stores the model at:

```text
/workspace/models/mobilenet_iter_73000.caffemodel
/workspace/models/mobilenet_ssd.prototxt
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

Limit the processing rate and skip frames if CPU load is high:

```bash
ros2 launch ros2_yolo_stream yolo_stream.launch.py \
  input_topic:=/image \
  output_topic:=/yolo/image \
  output_fps:=5.0 \
  max_processing_fps:=2.0 \
  process_every_n:=1 \
  output_reliability:=reliable \
  confidence_threshold:=0.35
```

Lower `output_fps` and `max_processing_fps` or increase `process_every_n` to
reduce CPU usage.
