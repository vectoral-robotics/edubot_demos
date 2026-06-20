from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("input_topic", default_value="/image"),
            DeclareLaunchArgument("output_topic", default_value="/yolo/image"),
            DeclareLaunchArgument("model_type", default_value="ssd"),
            DeclareLaunchArgument(
                "model_path", default_value="/workspace/models/mobilenet_iter_73000.caffemodel"
            ),
            DeclareLaunchArgument(
                "config_path", default_value="/workspace/models/mobilenet_ssd.prototxt"
            ),
            DeclareLaunchArgument("input_size", default_value="320"),
            DeclareLaunchArgument("confidence_threshold", default_value="0.35"),
            DeclareLaunchArgument("nms_threshold", default_value="0.45"),
            DeclareLaunchArgument("process_every_n", default_value="1"),
            DeclareLaunchArgument("max_processing_fps", default_value="2.0"),
            DeclareLaunchArgument("output_fps", default_value="5.0"),
            DeclareLaunchArgument("output_reliability", default_value="reliable"),
            DeclareLaunchArgument("watchdog_timeout_sec", default_value="5.0"),
            Node(
                package="ros2_yolo_stream",
                executable="yolo_stream_node",
                name="yolo_stream",
                output="screen",
                parameters=[
                    {
                        "input_topic": LaunchConfiguration("input_topic"),
                        "output_topic": LaunchConfiguration("output_topic"),
                        "model_type": LaunchConfiguration("model_type"),
                        "model_path": LaunchConfiguration("model_path"),
                        "config_path": LaunchConfiguration("config_path"),
                        "input_size": LaunchConfiguration("input_size"),
                        "confidence_threshold": LaunchConfiguration("confidence_threshold"),
                        "nms_threshold": LaunchConfiguration("nms_threshold"),
                        "process_every_n": LaunchConfiguration("process_every_n"),
                        "max_processing_fps": LaunchConfiguration("max_processing_fps"),
                        "output_fps": LaunchConfiguration("output_fps"),
                        "output_reliability": LaunchConfiguration("output_reliability"),
                        "watchdog_timeout_sec": LaunchConfiguration("watchdog_timeout_sec"),
                    }
                ],
            ),
        ]
    )
