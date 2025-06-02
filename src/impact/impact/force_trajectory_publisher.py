import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16, Float32
from geometry_msgs.msg import WrenchStamped
from impact_msgs.msg import GoalForceController

import numpy as np

import h5py
from collections import deque


class ForceTrajectoryPublisher(Node):

    def __init__(self):
        """
        This node publishes the goal force of the gripper in a simple linear trajectory.

        :return: None
        """

        super().__init__("force_trajectory_publisher")

        # load calibration parameters for gripper
        self.m, self.c = np.load("/home/erik/impact/src/actuated_umi/calibration/20250526-133247.npy")

        # load force trajectory from h5df file
        self.file = h5py.File("/home/erik/impact/src/impact/hdf5_data/assemble_task/rosbag2_2025_05_16-13_41_32_0.hdf5", "r")
        self.force_trajectory = list(self.file["feats/feats_fz"][:])
        self.position_trajectory = list(self.file["realsense_d405/realsense_d405_aruco_dist"][:])

        # create publisher to set goal force of gripper
        self.goal_force_publisher = self.create_publisher(GoalForceController, "set_actuated_umi_goal_force", 10)

        # create timer to control the force of the gripper
        self.control_frequency = 25
        self.force_goal_timer = self.create_timer(1.0 / self.control_frequency, self.pub_force_goal)


    def pub_force_goal(self):
        """
        Publish the goal force trajectory to the gripper.

        :return: None
        """
        
        # check if force trajectory is not empty
        if len(self.force_trajectory) > 0:
            
            # publish goal force
            msg = GoalForceController()
            msg.goal_force = float(self.force_trajectory.pop(0))
            msg.goal_position = int(self.m * self.position_trajectory.pop(0) + self.c)
            self.goal_force_publisher.publish(msg)
        

def main(args=None):
    """
    ROS node for the force trajectory publisher.

    :param args: arguments for the ROS node
    :return: None
    """

    try:
        
        print("Force Trajectory Publisher node is running... Press <ctrl> <c> to stop. \n")

        rclpy.init(args=args)

        force_trajectory_pub = ForceTrajectoryPublisher()

        rclpy.spin(force_trajectory_pub)

    finally:

        # Destroy the node explicitly
        # (optional - otherwise it will be done automatically
        # when the garbage collector destroys the node object)
        force_trajectory_pub.destroy_node()
        rclpy.shutdown()
    

if __name__ == "__main__":

    main()