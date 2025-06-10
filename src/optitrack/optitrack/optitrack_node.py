import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from scipy.spatial.transform import Rotation as R
import numpy as np

from std_msgs.msg import Float32MultiArray, Header
from geometry_msgs.msg import TransformStamped

from optitrack.optitrack import OptiTrack


class OptitrackNode(Node):

    def __init__(self):
        """
        This node is responsible for publishing the current position and orientation of the end-effector. 
        
        :return: None
        """

        super().__init__("optitrack_node")

        self.ot = OptiTrack()

        # get transformation matrix
        self.transformation = self.ot.transformation

        # create publisher for end-effector position and orientation
        self.optitrack_publisher = self.create_publisher(TransformStamped, "optitrack_ee_state", 1)
        timer_period = 0.02  # 50 Hz
        self.timer = self.create_timer(timer_period, self.get_current_state)


    def __del__(self):
        """
        Destructor to clean up resources.
        
        return: None
        """

        self.ot.close()


    def get_current_state(self):
        """
        Retrieve the current state of the end-effector and publish it.

        :return: None
        """

        # get the current state from the OptiTrack instance
        ee_pos = self.ot.ee_pos
        ee_ori = self.ot.ee_ori

        # extract rotation from transformation matrix
        R_T = self.transformation[:3, :3]

        # convert to quaternion
        q_T = R.from_matrix(R_T).as_quat()

        # Rotation transformieren
        rot_T = R.from_quat(q_T)
        rot_ee = R.from_quat(ee_ori)
        ee_ori_franka = (rot_T * rot_ee).as_quat()

        # transform end-effector position
        ee_pos_franka = self.transformation[:3, :3] @ ee_pos + self.transformation[:3, 3]

        print(ee_pos_franka)


def main(args=None):
    """
    ROS node for the OptiTrack system.

    :param args: arguments for the ROS node
    :return: None
    """

    try:

        print("Optitrack ROS node is running... Press <ctrl> <c> to stop. \nEndeffector state is being published on topic /optitrack_ee_state. \n")

        rclpy.init(args=args)

        optitrack_node = OptitrackNode()

        rclpy.spin(optitrack_node)

    finally:

        # Destroy the node explicitly
        # (optional - otherwise it will be done automatically
        # when the garbage collector destroys the node object)
        optitrack_node.__del__()
        optitrack_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    
    main()