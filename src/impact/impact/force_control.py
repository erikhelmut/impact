import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16, Float32
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import JointState
from impact_msgs.msg import GoalForceController
from collections import deque


class ForceControl(Node):

    def __init__(self):
        """
        This node controls the force of the gripper by setting the goal position of the motor.

        :return: None
        """
        
        super().__init__("force_control")

        # define control parameters
        self.kp = 77  # proportional gain
        self.kd = 10  # derivative gain
        self.alpha = 0.8  # low-pass filter parameter

        # store current force and goal force
        self.current_force = 0
        self.force_rate_filtered = 0
        self.goal_force = None

        # store previous force and force rate
        self.prev_force = 0
        self.prev_force_rate_filtered = 0

        # store current position of gripper
        self.current_position = None

        # create publisher to set goal position of gripper
        self.goal_position_publisher = self.create_publisher(Int16, "set_actuated_umi_motor_position", 10)

        # create subscriber to get current position of gripper
        self.current_position_subscriber = self.create_subscription(JointState, "actuated_umi_motor_state", self.get_current_position, 10)

        # create subscriber to get current force of gripper without callback
        self.current_force_subscriber = self.create_subscription(WrenchStamped, "resense_0", self.receive_force, 10)
        self.current_force_subscriber  # prevent unused variable warning
        # timer to process messages at 25 Hz (40 ms interval)
        self.resense_timer = self.create_timer(1.0 / 25, self.get_current_force)
        # buffer to store incoming messages
        self.message_queue = deque(maxlen=1)

        # create subscriber to set goal force of gripper
        self.goal_force_subscriber = self.create_subscription(GoalForceController, "set_actuated_umi_goal_force", self.set_goal_force, 10)
        self.goal_force_subscriber  # prevent unused variable warning

        # create timer to control the force of the gripper
        self.control_frequency = 25
        self.force_control_timer = self.create_timer(1.0 / self.control_frequency, self.force_control)


    def set_goal_position(self, position):
        """
        Set the goal motor position of the gripper.

        :param position: goal position of gripper
        :return: None
        """

        msg = Int16()
        msg.data = int(position)
        self.goal_position_publisher.publish(msg)


    def get_current_position(self, msg):
        """
        Get the current motor position of the gripper.

        :param msg: joint state message containing the current motor position
        :return: None
        """

        self.current_position = msg.position[0]

    
    def set_goal_force(self, msg):
        """
        Set the goal normal force of the gripper.

        :param msg: wrench message containing the goal normal force
        :return: None
        """

        self.goal_force = msg.goal_force
        self.kp = msg.kp
        self.kd = msg.kd
        self.alpha = msg.alpha


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

            # get latest message
            msg = self.message_queue.popleft()

            # store previous and current force
            self.prev_force = self.current_force
            self.current_force = msg.wrench.force.z

            # store previous force rate
            self.prev_force_rate_filtered = self.force_rate_filtered

            # calculate force rate (raw derivative)
            force_rate_unfiltered = (self.current_force - self.prev_force) / (1.0 / self.control_frequency)

            # apply low-pass filter to force rate and store it
            self.force_rate_filtered = self.alpha * self.prev_force_rate_filtered + (1 - self.alpha) * force_rate_unfiltered


    def force_control(self):
        """
        Control the force of the gripper via force feedback.

        :return: None
        """

        # control force if force control is active
        if self.goal_force is not None and self.current_position is not None:

            # calculate force error
            force_error = self.goal_force - self.current_force
            print("Force error: ", force_error)

            # compute relative position adjustment using PD control
            position_adjustment = self.kp * force_error + self.kd * self.force_rate_filtered

            # apply saturation (to avoid excessive movements)
            max_step = 42
            position_adjustment = max(min(position_adjustment, max_step), -max_step)

            # update gripper position
            self.set_goal_position(self.current_position + position_adjustment)


def main(args=None):
    """
    ROS node for the force control of the gripper.

    :param args: arguments for the ROS node
    :return: None
    """

    try:
        
        print("Force Control node is running... Press <ctrl> <c> to stop. \n")

        rclpy.init(args=args)

        force_control = ForceControl()

        rclpy.spin(force_control)

    finally:

        # Destroy the node explicitly
        # (optional - otherwise it will be done automatically
        # when the garbage collector destroys the node object)
        force_control.destroy_node()
        rclpy.shutdown()
    

if __name__ == "__main__":

    main()