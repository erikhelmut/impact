import rclpy
from rclpy.node import Node

from impact_interfaces.msg import ArUcoDistStamped
from std_msgs.msg import Int16

import time
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


class Imitate(Node):

    def __init__(self):

        super().__init__("imitate")

        # create publisher to set goal position of gripper
        self.publisher = self.create_publisher(Int16, "set_actuated_umi_motor_position", 10)


    def set_goal_position(self, position):
        """
        Set the goal position of the gripper.

        :param position: goal position of gripper
        :return: None
        """

        msg = Int16()
        msg.data = int(position)
        self.publisher.publish(msg)




def main(args=None):

    try:

        rclpy.init(args=args)

        imitator = Imitate()

        m, c = np.load("../calibration/20250304-134647.npy")
        print(m, c)

        reader = rosbag2_py.SequentialReader()

        # Define storage options
        storage_options = rosbag2_py.StorageOptions(
            uri="../bag_files/rosbag2_2025_03_04-12_40_45",
            storage_id="mcap"
        )

        # Define converter options
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr"
        )

        # Open the bag file correctly
        reader.open(storage_options, converter_options)

        # Get topic info
        topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}

        # Read messages and convert them
        distances_motor = []

        while reader.has_next():

            topic, data, timestamp = reader.read_next()
            
            msg_type = topic_types.get(topic, None)
            if msg_type:
                msg_class = get_message(msg_type)
                msg = deserialize_message(data, msg_class)
                print(f"Topic: {topic}, Timestamp: {timestamp}, Data: {msg}")
                print(msg.distance)
                distances_motor.append(m * msg.distance + c)

        i = 0
        while rclpy.ok():
            rclpy.spin_once(imitator, timeout_sec=0.04) # 0.04
            goal_position = distances_motor[i]
            imitator.set_goal_position(goal_position)
            print("\rGoal position:", goal_position)
            i += 1

            if i == len(distances_motor):
                break

            

    finally:
        imitator.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

    

