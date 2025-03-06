import argparse

import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py
from aruco_msgs.msg import ArUcoDistStamped
from std_msgs.msg import Int16

import numpy as np


class Imitator(Node):

    def __init__(self, calibration):
        """
        This node is responsible for imitating the gripper demonstration. It publishes the goal position of the gripper based on the aruco distance.

        :param calibration: path to the calibration file
        :return: None
        """

        super().__init__("imitator")

        # create publisher to set goal position of gripper
        self.publisher = self.create_publisher(Int16, "set_actuated_umi_motor_position", 10)

        # load calibration parameters for gripper
        self.m, self.c = np.load(calibration)

        # create storage options for goal distances
        self.goal_distances = []


    def set_goal_distance(self, distance):
        """
        Set goal distance between aruco markers. Convert distance to motor position using calibration parameters.

        :param distance: goal distance between aruco markers
        :return: None
        """

        msg = Int16()
        msg.data = int(self.m * distance + self.c)
        self.publisher.publish(msg)


def main(args=None):
    """
    Imitation script for the gripper demonstration.

    :param args: arguments for the ROS node
    :return: None
    """

    try:

        print("""
        ██╗███╗   ███╗██╗████████╗ █████╗ ████████╗ ██████╗ ██████╗ 
        ██║████╗ ████║██║╚══██╔══╝██╔══██╗╚══██╔══╝██╔═══██╗██╔══██╗
        ██║██╔████╔██║██║   ██║   ███████║   ██║   ██║   ██║███████║
        ██║██║╚██╔╝██║██║   ██║   ██╔══██║   ██║   ██║   ██║██╔═██╔╝ 
        ██║██║ ╚═╝ ██║██║   ██║   ██║  ██║   ██║   ╚██████╔╝██║  ██╗   
        ╚═╝╚═╝     ╚═╝╚═╝   ╚═╝   ╚═╝  ╚═╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝   
        """)

        print("Imitating the demonstration... \n")

        rclpy.init()

        # create imitator node
        imitator = Imitator(calibration=args.calibration)

        # create reader object
        reader = rosbag2_py.SequentialReader()

        # define storage options for bag file
        storage_options = rosbag2_py.StorageOptions(
            uri=args.bag_file,
            storage_id="mcap"
        )

        # define converter options
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr"
        )

        # open the bag file correctly
        reader.open(storage_options, converter_options)

        # get topic info
        topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}

        # read the bag file
        while reader.has_next():

            topic, data, timestamp = reader.read_next()
            msg_type = topic_types.get(topic, None)

            if topic == "/aruco_distance":
                msg_class = get_message(msg_type)
                msg = deserialize_message(data, msg_class)
                imitator.goal_distances.append(msg.distance)

        # replay the demonstration
        while rclpy.ok():
            rclpy.spin_once(imitator, timeout_sec=1/30)  # 30 Hz
            goal_distance = imitator.goal_distances.pop(0)
            imitator.set_goal_distance(goal_distance)
            print("Goal ArUco Distance:", goal_distance, end="\r", flush=True)

            # check if the demonstration is finished
            if not imitator.goal_distances:
                break

    finally:

        # Destroy the node explicitly
        # (optional - otherwise it will be done automatically
        # when the garbage collector destroys the node object)
        imitator.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    
    # parse arguments
    parser = argparse.ArgumentParser(description="Imitate the gripper demonstration.")
    parser.add_argument("--calibration", type=str, default="../../actuated_umi/calibration/20250304-134647.npy", help="Path to the calibration file")
    parser.add_argument("--bag_file", type=str, default="../bag_files/rosbag2_2025_03_06-13_39_24", help="Path to the bag file")
    args = parser.parse_args()

    main(args)

    

