import argparse

import rclpy
from rclpy.node import Node
from message_filters import Subscriber, ApproximateTimeSynchronizer
from std_msgs.msg import Int16
from sensor_msgs.msg import JointState
from impact_interfaces.msg import ArUcoDistStamped

import numpy as np
import time
import matplotlib.pyplot as plt


class CalibrateGripper(Node):

    def __init__(self):
        """
        This node is responsible for calibrating the gripper. It subscribes to the aruco distance and motor state topics and creates a least squares fit to determine the relationship between the motor position and the aruco distance.

        :return: None
        """

        super().__init__("calibrate_gripper")

        # create subscriber for aruco distance
        self.subscriber_aruco_dist = Subscriber(self, ArUcoDistStamped, "aruco_distance")

        # create subscriber for motor state
        self.subscriber_motor_state = Subscriber(self, JointState, "actuated_umi_motor_state")

        # create publisher to set goal position of gripper
        self.publisher = self.create_publisher(Int16, "set_actuated_umi_motor_position", 10)

        # set start position of gripper
        self.set_goal_position(900)

        # save msgs for aruco distance and motor position
        self.aruco_dist = []
        self.motor_pos = []

        # create synchronizer
        self.sync = ApproximateTimeSynchronizer(
            [self.subscriber_aruco_dist, self.subscriber_motor_state], queue_size=10, slop=0.1  # 100 ms
        )
        self.sync.registerCallback(self.sync_callback)


    def sync_callback(self, msg_aruco_dist, msg_motor_state):
        """
        Synchronizer callback function to save the aruco distance and motor position.

        :param msg_aruco_dist: ArUcoDistStamped message
        :param msg_motor_state: JointState message
        :return: None
        """

        self.aruco_dist.append(msg_aruco_dist.distance)
        self.motor_pos.append(msg_motor_state.position[0])


    def set_goal_position(self, position):
        """
        Set the goal position of the gripper.

        :param position: goal position of gripper
        :return: None
        """

        msg = Int16()
        msg.data = int(position)
        self.publisher.publish(msg)


def main(folder, args=None):
    """
    Calibration script for the gripper.

    :param folder: folder to save calibration data
    :param args: arguments for the ROS node
    :return: None
    """

    try:
        
        print("""
            ________________________
        <--| CALIBRATE ACTUATED UMI |-->
            ------------------------
        """)

        print("Calibrating the actuated-UMI gripper... \n")

        rclpy.init(args=args)

        calibrate_gripper = CalibrateGripper()

        goal_position = 800

        while rclpy.ok():
            rclpy.spin_once(calibrate_gripper, timeout_sec=1.0)

            # call every 0.1 seconds set_goal_position
            current_time = time.time()
            if not hasattr(calibrate_gripper, 'last_call_time'):
                calibrate_gripper.last_call_time = current_time

            if current_time - calibrate_gripper.last_call_time >= 0.1:
                calibrate_gripper.set_goal_position(goal_position)
                goal_position -= 10
                calibrate_gripper.last_call_time = current_time

            if goal_position < -2380:
                break
        
        # least squares fit to determine the relationship between the aruco distance and motor position
        m, c = np.polyfit(calibrate_gripper.aruco_dist, calibrate_gripper.motor_pos, 1)
        
        np.save(folder + time.strftime("%Y%m%d-%H%M%S") + ".npy", [m, c])

        # plot motor position vs aruco distance
        plt.scatter(calibrate_gripper.aruco_dist, calibrate_gripper.motor_pos)
        plt.plot(calibrate_gripper.aruco_dist, m * np.array(calibrate_gripper.aruco_dist) + c, color="red")
        plt.ylabel("Motor Position")
        plt.xlabel("ArUco Distance")
        plt.title("Motor Position vs ArUco Distance")
        plt.show()

        calibrate_gripper.set_goal_position(1000)
        print("Calibration completed. \n")

    finally:
        
        # Destroy the node explicitly
        # (optional - otherwise it will be done automatically
        # when the garbage collector destroys the node object)
        calibrate_gripper.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="../calibration/", help="Folder to save calibration data")
    args = parser.parse_args()

    main(folder=args.folder)