import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    actuated_umi_share_dir = get_package_share_directory("actuated_umi")
    actuated_umi_config_file = os.path.join(actuated_umi_share_dir, "config", "inference_qos.yaml")

    gelsight_mini_share_dir = get_package_share_directory("gelsight_mini")
    gelsight_mini_config_file = os.path.join(gelsight_mini_share_dir, "config", "inference_qos.yaml")

    feats_share_dir = get_package_share_directory("feats")
    feats_config_file = os.path.join(feats_share_dir, "config", "inference_qos.yaml")

    realsense_d405_share_dir = get_package_share_directory("realsense_d405")
    realsense_d405_config_file = os.path.join(realsense_d405_share_dir, "config", "inference_qos.yaml")

    optitrack_share_dir = get_package_share_directory("optitrack")
    optitrack_config_file = os.path.join(optitrack_share_dir, "config", "inference_qos.yaml")

    return LaunchDescription([
        Node(
            package="actuated_umi",
            executable="actuated_umi_node",
            name="actuated_umi_node",
            parameters=[actuated_umi_config_file],
            output="screen"
        ),
        Node(
            package="gelsight_mini",
            executable="gelsight_mini_node",
            name="gelsight_mini_node",
            parameters=[gelsight_mini_config_file],
            output="screen"
        ),
        Node(
            package="feats",
            executable="feats_node",
            name="feats_node",
            parameters=[feats_config_file],
            output="screen"
        ),
        Node(
            package="realsense_d405",
            executable="realsense_d405_node",
            name="realsense_d405_node",
            parameters=[realsense_d405_config_file],
            output="screen"
        ),
        Node(
            package="realsense_d405",
            executable="detect_aruco_node",
            name="detect_aruco_node",
            parameters=[realsense_d405_config_file],
            output="screen"
        ),
        Node(
            package="optitrack",
            executable="optitrack_node",
            name="optitrack_node",
            parameters=[optitrack_config_file],
            output="screen"
        ),
    ])