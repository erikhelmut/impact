import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import Int16, Float32
from geometry_msgs.msg import WrenchStamped
from feats_msgs.msg import ForceDistStamped
from sensor_msgs.msg import JointState
from impact_msgs.msg import GoalForceController
from collections import deque
import copy


class MovingAverage:

    def __init__(self, size):
        self.window = deque(maxlen=size)


    def filter(self, value):
        self.window.append(value)
        return sum(self.window) / len(self.window)


class ForceControl(Node):

    def __init__(self):
        """
        This node controls the force of the gripper.

        :return: None
        """
        
        super().__init__("force_control")

        # define control parameters
        self.kp = 25  # proportional gain
        self.kd = 0  # derivative gain
        self.alpha = 0.8  # low-pass filter parameter

        # store current position, force and goal force
        self.current_position = None
        self.current_force = 0
        self.force_rate_filtered = 0
        self.goal_force = None

        # store previous force and force rate
        self.prev_force = 0
        self.prev_force_rate_filtered = 0

        # store time of previous and current message
        self.prev_time = 0
        self.current_time = 0

        # moving average filter for force
        self.filt = MovingAverage(size=10)

        # create publisher to set goal position of gripper
        self.goal_position_publisher = self.create_publisher(Int16, "set_actuated_umi_motor_position", 10)

        # create subscriber to get current position of gripper
        self.current_position_subscriber = self.create_subscription(JointState, "actuated_umi_motor_state", self.get_current_position, 10)

        # create subscriber to get current force of gripper without callback
        current_force_subscriber_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.current_force_subscriber = self.create_subscription(ForceDistStamped, "feats_fz", self.get_current_force, current_force_subscriber_qos_profile)
        self.current_force_subscriber  # prevent unused variable warning

        # create publisher for filtered force
        ma_force_publisher_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.filtered_force_publisher = self.create_publisher(Float32, "feats_ma_fz", ma_force_publisher_qos_profile)

        # create subscriber to set goal force of gripper
        self.goal_force_subscriber = self.create_subscription(GoalForceController, "set_actuated_umi_goal_force", self.set_goal_force, 10)
        self.goal_force_subscriber  # prevent unused variable warning

        # create timer to control the force of the gripper
        self.control_frequency = 50
        self.force_control_timer = self.create_timer(1.0 / self.control_frequency, self.force_control)

        
    def set_goal_position(self, position_adjustment):
        """
        Set the goal position of the gripper.

        :param position: goal position of gripper
        :return: None
        """

        msg = Int16()
        msg.data = int(self.current_position + position_adjustment)
        self.goal_position_publisher.publish(msg)


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


    def get_current_position(self, msg):
        """
        Get the current position of the gripper.
        
        :param msg: message containing the current position
        :return: None
        """

        self.current_position = msg.position[0]


    def get_current_force(self, msg):
        """
        Get the current normal force of the gripper.
        Calculate the force rate and apply a low-pass filter to it.

        :return: None
        """

        # store previous
        self.prev_force = copy.copy(self.current_force)
        self.prev_time = copy.copy(self.current_time)

        # store previous force rate
        self.prev_force_rate_filtered = copy.copy(self.force_rate_filtered)
        
        # store current force
        self.current_force = self.filt.filter(msg.f)
        self.current_time = self.get_clock().now()
        self.current_time = self.current_time.nanoseconds / 1e9  # convert to seconds

        # calculate force rate (raw derivative)
        force_rate_unfiltered = (self.current_force - self.prev_force) / (self.current_time - self.prev_time)

        # apply low-pass filter to force rate and store it
        self.force_rate_filtered = self.alpha * self.prev_force_rate_filtered + (1 - self.alpha) * force_rate_unfiltered

        # publish filtered force
        filtered_force = Float32()
        filtered_force.data = self.current_force
        self.filtered_force_publisher.publish(filtered_force)


    def force_control(self):
        """
        Control the force of the gripper via force feedback.

        :return: None
        """

        if self.goal_force is not None and self.current_position is not None:

            # calculate force error
            force_error = self.goal_force - self.current_force
            print("Force error: ", force_error)

            # compute position adjustment
            position_adjustment = self.kp * force_error + self.kd * self.force_rate_filtered

            # apply saturation (to avoid excessive movements)
            rel_position_limit = 50
            position_adjustment = max(min(position_adjustment, rel_position_limit), -rel_position_limit)

            # update goal position
            self.set_goal_position(position_adjustment)


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