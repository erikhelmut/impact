import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16, Header
from sensor_msgs.msg import JointState

from dynamixel_api import XL430W250TConnector
from actuated_umi.motor_control import ActuatedUMI


class ActuatedUMINode(Node):

    def __init__(self, device="/dev/ttyUSB0"):
        """
        This node is responsible for controlling the actuated-UMI in extended position control mode. It publishes the current motor state of the gripper and subscribes to a topic to set the goal position of the gripper.
        
        :param device: device to which the gripper is connected
        :return: None
        """

        super().__init__("actuated_umi_node")

        # setup connector for the Dynamixel XL430-W250-T
        self.connector = XL430W250TConnector(device=device, baud_rate=57600, dynamixel_id=1)
        self.connector.connect()

        # connect to the gripper
        self.gripper = ActuatedUMI(self.connector)
        self.gripper.torque_enabled = False
        self.gripper.operating_mode = 4  # extended position control
        
        # set the PID gains
        self.gripper.position_p_gain = 640
        self.gripper.position_i_gain = 0
        self.gripper.position_d_gain = 3600

        # enable torque
        self.gripper.torque_enabled = True

        # create publisher for current position of gripper
        self.publisher_ = self.create_publisher(JointState, "actuated_umi_motor_state", 10)
        timer_period = 0.02  # 50 Hz
        self.timer = self.create_timer(timer_period, self.get_current_state)

        # create subscriber to set goal position of gripper
        self.subscriber = self.create_subscription(Int16, "set_actuated_umi_motor_position", self.set_goal_position, 10)
        self.subscriber  # prevent unused variable warning


    def __del__(self):
        """
        Destructor for the actuated UMI node.

        :return: None
        """

        self.gripper.torque_enabled = False
        self.connector.disconnect()


    def set_goal_position(self, msg):
        """
        Set the goal motor position of the gripper.

        :param msg: message containing the goal position
        :return: None
        """

        # check if the goal position is within the limits
        if msg.data >= -2380:
            self.gripper.goal_position = msg.data


    def get_current_state(self):
        """
        Callback function to publish the current motor state of the gripper.

        :return: None
        """

        header = Header()
        header.stamp = self.get_clock().now().to_msg()

        msg = JointState()
        msg.header = header
        msg.name = ["Dynamixel XL430-W250-T"]
        msg.position = np.array([self.gripper.current_position])
        msg.velocity = np.array([self.gripper.current_velocity])
        msg.effort = np.array([self.gripper.current_load])
        self.publisher_.publish(msg)


def main(args=None):
    """
    ROS node for the actuated-UMI gripper.

    :param args: arguments for the ROS node
    :return: None
    """

    try:

        print("""
        ACTUATED UMI
          _______
         |  O O  |
         |_______|
           ||||      
           ||||      
        \\  |  |  /   
         \\ |  | /    
          \\|  |/     
        """)
        
        print("Actuated UMI ROS node is running... Press <ctrl> <c> to stop. \nMotor state is being published on topic /actuated_umi_motor_state. \n")

        rclpy.init(args=args)

        actuated_umi_node = ActuatedUMINode()

        rclpy.spin(actuated_umi_node)
        
    finally:
        
        # Destroy the node explicitly
        # (optional - otherwise it will be done automatically
        # when the garbage collector destroys the node object)
        actuated_umi_node.__del__()
        actuated_umi_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    
    main()