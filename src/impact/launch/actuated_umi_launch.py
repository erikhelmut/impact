from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='actuated_umi',
            executable='actuated_umi_node',
        ),
        Node(
            package='realsense_d405',
            executable='realsense_d405_node',
        ),
        Node(
            package='realsense_d405',
            executable='detect_aruco_node',
        ),
        Node(
            package='gelsight_mini',
            executable='gelsight_mini_node',
        ),
        Node(
            package='resense_hex21',
            executable='resense_hex21_node',
        )
    ])