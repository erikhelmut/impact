import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import Int16, Header
from sensor_msgs.msg import JointState

import time

from dynamixel_api import XL430W250TConnector
from actuated_umi.motor_control import ActuatedUMI

import numpy as np


class ActuatedUMINode(Node):

    def __init__(self, device="/dev/ttyUSB0"):
        """
        This node is responsible for controlling the actuated-UMI. It publishes the current motor state of the gripper and subscribes to a topic to set the goal position, goal velocity or goal pwm of the gripper.
        
        :param device: device to which the gripper is connected
        :return: None
        """

        super().__init__("actuated_umi_node")

        # setup connector for the Dynamixel XL430-W250-T
        self.connector = XL430W250TConnector(device=device, baud_rate=2000000, dynamixel_id=1)
        self.connector.connect()

        # connect to the gripper
        self.gripper = ActuatedUMI(self.connector)
        self.gripper.torque_enabled = False
        self.gripper.operating_mode = 4  # extended position control mode
        
        # set the PID gains for position control
        self.gripper.position_p_gain = 640
        self.gripper.position_i_gain = 0
        self.gripper.position_d_gain = 4000

        # set the PI gains for velocity control
        self.gripper.velocity_p_gain = 100
        self.gripper.velocity_i_gain = 1000

        # enable torque
        self.gripper.torque_enabled = True

        # declare parameters for QoS settings
        self.declare_parameter("actuated_umi.qos.reliability", "reliable")
        self.declare_parameter("actuated_umi.qos.history", "keep_last")
        self.declare_parameter("actuated_umi.qos.depth", 10)

        # get QoS profile
        qos_profile = self.get_qos_profile("actuated_umi.qos")

        # create publisher for current position of gripper
        self.publisher_ = self.create_publisher(JointState, "actuated_umi_motor_state", qos_profile)
        timer_period = 0.02  # 50 Hz
        self.timer = self.create_timer(timer_period, self.get_current_state)

        # create subscriber to set goal position of gripper
        self.goal_position_subscriber = self.create_subscription(Int16, "set_actuated_umi_motor_position", self.set_goal_position, qos_profile)
        self.goal_position_subscriber  # prevent unused variable warning

        # create subscriber to set goal velocity of gripper
        self.goal_velocity_subscriber = self.create_subscription(Int16, "set_actuated_umi_motor_velocity", self.set_goal_velocity, qos_profile)
        self.goal_velocity_subscriber  # prevent unused variable warning

        # create subscriber to set goal PWM of gripper
        self.goal_pwm_subscriber = self.create_subscription(Int16, "set_actuated_umi_motor_pwm", self.set_goal_pwm, qos_profile)
        self.goal_pwm_subscriber  # prevent unused variable warning


    def __del__(self):
        """
        Destructor for the actuated UMI node.

        :return: None
        """

        self.gripper.torque_enabled = False
        time.sleep(0.2)
        self.gripper.__del__()
        self.connector.disconnect()


    def get_qos_profile(self, base_param_name):
        """
        Helper function to retrieve and validate QoS settings.
        
        :param base_param_name: base name of the QoS parameters
        :return: QoSProfile object
        """
        
        # get the parameter values
        reliability_param = self.get_parameter(f"{base_param_name}.reliability").value
        history_param = self.get_parameter(f"{base_param_name}.history").value
        depth_param = self.get_parameter(f"{base_param_name}.depth").value

        # normalize to lowercase to avoid mismatches
        reliability_param = str(reliability_param).lower()
        history_param = str(history_param).lower()

        self.get_logger().info(f"QoS settings: reliability={reliability_param}, history={history_param}, depth={depth_param}")

        # convert to QoS enums with fallback
        if reliability_param == "best_effort":
            reliability = QoSReliabilityPolicy.BEST_EFFORT
        elif reliability_param == "reliable":
            reliability = QoSReliabilityPolicy.RELIABLE
        else:
            self.get_logger().warn(f"Unknown reliability: {reliability_param}, defaulting to RELIABLE")
            reliability = QoSReliabilityPolicy.RELIABLE

        if history_param == "keep_last":
            history = QoSHistoryPolicy.KEEP_LAST
        elif history_param == "keep_all":
            history = QoSHistoryPolicy.KEEP_ALL
        else:
            self.get_logger().warn(f"Unknown history: {history_param}, defaulting to KEEP_LAST")
            history = QoSHistoryPolicy.KEEP_LAST

        # depth should be an int, just check type or cast
        try:
            depth = int(depth_param)
        except (ValueError, TypeError):
            self.get_logger().warn(f"Invalid depth: {depth_param}, defaulting to 10")
            depth = 10

        # return the QoSProfile
        return QoSProfile(
            reliability=reliability,
            history=history,
            depth=depth
        )


    def set_goal_position(self, msg):
        """
        Set the goal motor position of the gripper.

        :param msg: message containing the goal position
        :return: None
        """

        # activate the gripper in extended position control mode
        if self.gripper.operating_mode != 4:
            self.gripper.torque_enabled = False
            self.gripper.operating_mode = 4
            self.gripper.torque_enabled = True

        # check if the goal position is within the limits
        if msg.data >= -2380:
            # TODO: try async write
            # self.gripper.__connector.write_async
            self.gripper.goal_position = msg.data

    
    def set_goal_velocity(self, msg):
        """
        Set the goal motor velocity of the gripper.

        :param msg: message containing the goal velocity
        :return: None
        """

        # activate the gripper in velocity control mode
        if self.gripper.operating_mode != 1:
            self.gripper.torque_enabled = False
            self.gripper.operating_mode = 1
            self.gripper.torque_enabled = True
        
        # check if the goal velocity is within the limits
        if msg.data >= -1023 and msg.data <= 1023:
            self.gripper.goal_velocity = msg.data


    def set_goal_pwm(self, msg):
        """
        Set the goal PWM of the gripper.

        :param msg: message containing the goal PWM value
        :return: None
        """

        # activate the gripper in PWM control mode
        if self.gripper.operating_mode != 16:
            self.gripper.torque_enabled = False
            self.gripper.operating_mode = 16
            self.gripper.torque_enabled = True

        # check if the pwm is between -885 and 885
        if msg.data >= -885 and msg.data <= 885:
            self.gripper.goal_pwm = msg.data


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
        #msg.velocity = np.array([self.gripper.current_velocity])
        #msg.effort = np.array([self.gripper.current_load])
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