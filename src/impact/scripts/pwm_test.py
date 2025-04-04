import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16
from geometry_msgs.msg import WrenchStamped
from collections import deque

import time
import matplotlib.pyplot as plt


class PWMCalibration(Node):

    def __init__(self):
        
        super().__init__("pwm_calibration")
        
        self.pwm = 0
        self.current_force = 0

        # create publisher to set goal pwm of gripper
        self.goal_pwm_publisher = self.create_publisher(Int16, "set_actuated_umi_motor_pwm", 10)

        # create subscribe to get goal pwm of gripper
        self.goal_pwm_subscriber = self.create_subscription(Int16, "get_actuated_umi_motor_goal_pwm", self.get_goal_pwm, 10)
        self.goal_pwm_subscriber  # prevent unused variable warning

        # create subscriber to get current force of gripper without callback
        self.current_force_subscriber = self.create_subscription(WrenchStamped, "resense_0", self.receive_force, 10)
        self.current_force_subscriber  # prevent unused variable warning
        # timer to process messages at 25 Hz (40 ms interval)
        self.resense_timer = self.create_timer(1.0 / 25, self.get_current_force)
        # buffer to store incoming messages
        self.message_queue = deque(maxlen=1)


    def set_goal_pwm(self, pwm):
        """
        Set the goal PWM value of the gripper.

        :param pwm: goal PWM value of gripper
        :return: None
        """

        self.pwm = pwm

        msg = Int16()
        msg.data = int(pwm)
        self.goal_pwm_publisher.publish(msg)

    
    def get_goal_pwm(self, msg):
        """
        Get the goal PWM value of the gripper.

        :param msg: goal pwm message
        :return: None
        """

        if self.pwm > msg.data:
            self.set_goal_pwm(msg.data)
        elif self.pwm < msg.data:
            self.set_goal_pwm(0)
            time.sleep(0.02)
            self.set_goal_pwm(msg.data)


    def receive_force(self, msg):
        """
        Receive the current force of the gripper.

        :param msg: wrench message containing the current normal force
        :return: None
        """

        self.message_queue.append(msg)


    def get_current_force(self):
        """
        Get the current normal force of the gripper in 25 Hz.
        Calculate the force rate and apply a low-pass filter to it.

        :return: None
        """

        if self.message_queue:
            msg = self.message_queue.popleft()
            self.current_force = msg.wrench.force.z


def main(args=None):
    """
    ROS node for the PWM calibration of the gripper.

    :param args: arguments for the ROS node
    :return: None
    """


    try:

        print("Starting PWM calibration node... Press <ctrl> <c> to stop. \n")

        rclpy.init(args=args)

        pwm_calibration = PWMCalibration()

        rclpy.spin(pwm_calibration)


    finally:

        # Destroy the node explicitly
        # (optional - otherwise it will be done automatically
        # when the garbage collector destroys the node object)
        pwm_calibration.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    
    main()
