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

        # create publisher to set goal position of gripper
        self.goal_pwm_publisher = self.create_publisher(Int16, "set_actuated_umi_motor_pwm", 10)

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

        force_values = []
        pwm_values = []

        i = 0

        pwm_calibration.set_goal_pwm(-120)

        while rclpy.ok():
            rclpy.spin_once(pwm_calibration, timeout_sec=1.0)


            if pwm_calibration.current_force < -0.5:
                pwm_calibration.set_goal_pwm(pwm_calibration.pwm)
                pwm_calibration.pwm -= 1

                time.sleep(1)

                # store current force and pwm values
                force_values.append(pwm_calibration.current_force)
                pwm_values.append(pwm_calibration.pwm)

                # print current force and pwm values
                print(f"Current PWM: {pwm_calibration.pwm}, Current Force: {pwm_calibration.current_force}")

                # stop after 150 PWM values
                if pwm_calibration.pwm <= -150:
                    break

        # plot the force vs pwm values
        plt.plot(pwm_values, force_values)
        plt.xlabel("PWM")
        plt.ylabel("Force (N)")
        plt.title("PWM vs Force")
        plt.grid()
        plt.show()


    finally:

        # Destroy the node explicitly
        # (optional - otherwise it will be done automatically
        # when the garbage collector destroys the node object)
        pwm_calibration.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    
    main()
