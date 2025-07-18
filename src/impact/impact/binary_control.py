import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import Int16, Float32
from feats_msgs.msg import ForceDistStamped
from sensor_msgs.msg import JointState
from impact_msgs.msg import GoalForceController

from collections import deque
import copy

from simple_pid import PID


class MovingAverage:
    """
    A simple moving average filter to smooth out the force readings.
    """

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

        # define PID controller 
        # (12, 1, 2) seems to work well
        self.pid = PID(10, 1, 2, setpoint=0.0, sample_time=None, starting_output=0.0, output_limits=(-80, 80))

        # store current position, force and goal force
        self.current_position = None
        self.goal_position = None
        self.current_force = 0
        self.goal_force = None

        # moving average filter for force
        self.filt = MovingAverage(size=10)

        # create publisher to set goal position of gripper
        self.goal_position_publisher = self.create_publisher(Int16, "set_actuated_umi_motor_position", 1)

        # create publisher to set pwm of gripper
        self.pwm = False
        self.goal_pwm_publisher = self.create_publisher(Int16, "set_actuated_umi_motor_pwm", 1)

        # create subscriber to get current position of gripper
        self.current_position_subscriber = self.create_subscription(JointState, "actuated_umi_motor_state", self.get_current_position, 1)

        # create subscriber to get current force of gripper without callback
        current_force_subscriber_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.current_force_subscriber = self.create_subscription(ForceDistStamped, "feats_fz", self.get_current_force, current_force_subscriber_qos_profile)
        self.current_force_subscriber  # prevent unused variable warning

        # create publisher for filtered force
        ma_force_publisher_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.filtered_force_publisher = self.create_publisher(Float32, "feats_ma_fz", ma_force_publisher_qos_profile)

        # create subscriber to set goal force of gripper
        self.goal_force_subscriber = self.create_subscription(GoalForceController, "set_actuated_umi_goal_force", self.set_goal_force, 1)
        self.goal_force_subscriber  # prevent unused variable warning

        
    def set_goal_position(self, position):
        """
        Set the goal position of the gripper.

        :param position: goal position of gripper
        :return: None
        """

        msg = Int16()
        msg.data = int(position)
        self.goal_position_publisher.publish(msg)


    def set_goal_force(self, msg):
        """
        Set the goal normal force of the gripper.

        :param msg: wrench message containing the goal normal force
        :return: None
        """

        #self.goal_force = 0.0
        self.goal_force = msg.goal_force
        self.goal_position = msg.goal_position


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

        # store current force
        #self.current_force = msg.f
        self.current_force = self.filt.filter(msg.f)
        
        # publish filtered force
        filtered_force = Float32()
        filtered_force.data = self.current_force
        self.filtered_force_publisher.publish(filtered_force)

        # call force control function
        self.force_control()


    def force_control(self):
        """
        Control the force of the gripper via force feedback.

        :return: None
        """

        if self.goal_force is not None and self.current_position is not None and self.goal_position is not None:

            if self.goal_position <= -400:
                # set pwm to -300
                if self.pwm == False:
                    self.goal_pwm_publisher.publish(Int16(data=-300))
                    self.pwm = True
            else:
                # disable pwm
                self.pwm = False
                self.goal_pwm_publisher.publish(Int16(data=7777))
                

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