from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("input_topic", default_value="/image"),
            DeclareLaunchArgument("output_topic", default_value="/follower/image"),
            DeclareLaunchArgument("cmd_vel_topic", default_value="/cmd_vel"),
            DeclareLaunchArgument(
                "model_path", default_value="/workspace/models/mobilenet_iter_73000.caffemodel"
            ),
            DeclareLaunchArgument(
                "config_path", default_value="/workspace/models/mobilenet_ssd.prototxt"
            ),
            DeclareLaunchArgument("confidence_threshold", default_value="0.40"),
            DeclareLaunchArgument("max_linear_speed", default_value="0.3"),
            DeclareLaunchArgument("max_angular_speed", default_value="0.8"),
            DeclareLaunchArgument("target_box_ratio", default_value="0.30"),
            DeclareLaunchArgument("dead_zone", default_value="0.05"),
            DeclareLaunchArgument("smoothing", default_value="0.15"),
            DeclareLaunchArgument("control_hz", default_value="20.0"),
            DeclareLaunchArgument("video_hz", default_value="10.0"),
            DeclareLaunchArgument("lost_timeout_sec", default_value="2.0"),
            DeclareLaunchArgument("search_speed", default_value="0.3"),
            DeclareLaunchArgument("enabled", default_value="true"),
            Node(
                package="ros2_person_follower",
                executable="person_follower_node",
                name="person_follower",
                output="screen",
                parameters=[
                    {
                        "input_topic": LaunchConfiguration("input_topic"),
                        "output_topic": LaunchConfiguration("output_topic"),
                        "cmd_vel_topic": LaunchConfiguration("cmd_vel_topic"),
                        "model_path": LaunchConfiguration("model_path"),
                        "config_path": LaunchConfiguration("config_path"),
                        "confidence_threshold": LaunchConfiguration("confidence_threshold"),
                        "max_linear_speed": LaunchConfiguration("max_linear_speed"),
                        "max_angular_speed": LaunchConfiguration("max_angular_speed"),
                        "target_box_ratio": LaunchConfiguration("target_box_ratio"),
                        "dead_zone": LaunchConfiguration("dead_zone"),
                        "smoothing": LaunchConfiguration("smoothing"),
                        "control_hz": LaunchConfiguration("control_hz"),
                        "video_hz": LaunchConfiguration("video_hz"),
                        "lost_timeout_sec": LaunchConfiguration("lost_timeout_sec"),
                        "search_speed": LaunchConfiguration("search_speed"),
                        "enabled": LaunchConfiguration("enabled"),
                    }
                ],
            ),
        ]
    )
