from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("input_topic", default_value="/image"),
            DeclareLaunchArgument("output_topic", default_value="/yolo/image"),
            DeclareLaunchArgument("model_path", default_value="/workspace/models/yolov5n.onnx"),
            DeclareLaunchArgument("input_size", default_value="320"),
            DeclareLaunchArgument("confidence_threshold", default_value="0.35"),
            DeclareLaunchArgument("nms_threshold", default_value="0.45"),
            DeclareLaunchArgument("process_every_n", default_value="3"),
            Node(
                package="ros2_yolo_stream",
                executable="yolo_stream_node",
                name="yolo_stream",
                output="screen",
                parameters=[
                    {
                        "input_topic": LaunchConfiguration("input_topic"),
                        "output_topic": LaunchConfiguration("output_topic"),
                        "model_path": LaunchConfiguration("model_path"),
                        "input_size": LaunchConfiguration("input_size"),
                        "confidence_threshold": LaunchConfiguration("confidence_threshold"),
                        "nms_threshold": LaunchConfiguration("nms_threshold"),
                        "process_every_n": LaunchConfiguration("process_every_n"),
                    }
                ],
            ),
        ]
    )
