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
            DeclareLaunchArgument("hsv_low_h", default_value="25"),
            DeclareLaunchArgument("hsv_low_s", default_value="40"),
            DeclareLaunchArgument("hsv_low_v", default_value="40"),
            DeclareLaunchArgument("hsv_high_h", default_value="95"),
            DeclareLaunchArgument("hsv_high_s", default_value="255"),
            DeclareLaunchArgument("hsv_high_v", default_value="255"),
            DeclareLaunchArgument("min_area", default_value="200"),
            DeclareLaunchArgument("max_linear_speed", default_value="0.3"),
            DeclareLaunchArgument("max_angular_speed", default_value="1.0"),
            DeclareLaunchArgument("target_radius_ratio", default_value="0.12"),
            DeclareLaunchArgument("dead_zone", default_value="0.04"),
            DeclareLaunchArgument("smoothing", default_value="0.20"),
            DeclareLaunchArgument("control_hz", default_value="20.0"),
            DeclareLaunchArgument("video_hz", default_value="15.0"),
            DeclareLaunchArgument("lost_timeout_sec", default_value="1.5"),
            DeclareLaunchArgument("search_speed", default_value="0.4"),
            DeclareLaunchArgument("enabled", default_value="true"),
            Node(
                package="ros2_color_follower",
                executable="color_follower_node",
                name="color_follower",
                output="screen",
                parameters=[
                    {
                        "input_topic": LaunchConfiguration("input_topic"),
                        "output_topic": LaunchConfiguration("output_topic"),
                        "cmd_vel_topic": LaunchConfiguration("cmd_vel_topic"),
                        "hsv_low_h": LaunchConfiguration("hsv_low_h"),
                        "hsv_low_s": LaunchConfiguration("hsv_low_s"),
                        "hsv_low_v": LaunchConfiguration("hsv_low_v"),
                        "hsv_high_h": LaunchConfiguration("hsv_high_h"),
                        "hsv_high_s": LaunchConfiguration("hsv_high_s"),
                        "hsv_high_v": LaunchConfiguration("hsv_high_v"),
                        "min_area": LaunchConfiguration("min_area"),
                        "max_linear_speed": LaunchConfiguration("max_linear_speed"),
                        "max_angular_speed": LaunchConfiguration("max_angular_speed"),
                        "target_radius_ratio": LaunchConfiguration("target_radius_ratio"),
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
