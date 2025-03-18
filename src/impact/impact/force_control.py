import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16, Float32
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import JointState



class ForceControl(Node):

    def __init__(self):
        
        super().__init__('force_control')

        # store current force and goal force
        self.current_force = None
        self.goal_force = None

        # store current position of gripper
        self.current_position = None

        # create publisher to set goal position of gripper
        self.goal_position_publisher = self.create_publisher(Int16, "set_actuated_umi_motor_position", 10)

        # create subscriber to get current position of gripper
        self.current_position_subscriber = self.create_subscription(JointState, "actuated_umi_motor_state", self.get_current_position, 10)

        # create subscriber to get current force of gripper
        self.current_force_subscriber = self.create_subscription(WrenchStamped, "resense_0", self.get_current_force, 10)
        self.current_force_subscriber  # prevent unused variable warning

        # create subscriber to set goal force of gripper
        self.goal_force_subscriber = self.create_subscription(Float32, "set_actuated_umi_goal_force", self.set_goal_force, 10)
        self.goal_force_subscriber  # prevent unused variable warning

        # create timer to control the force of the gripper
        self.timer = self.create_timer(0.01, self.force_control)


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

        self.goal_force = msg.data


    def get_current_force(self, msg):
        """
        Get the current normal force of the gripper.

        :param msg: wrench message containing the current normal force
        :return: None
        """

        self.current_force = msg.wrench.force.z 


    def force_control(self):

        kp = 5  # probably to little
        kd = 0.1
        prev_error = 0

        if self.current_force is not None and self.goal_force is not None:
            
            # control error
            error = self.goal_force - self.current_force

            derivative = (error - prev_error) / 0.01

            command = kp * error + kd * derivative
            
            prev_error = error

            print("Error: ", error)

            # set goal position
            self.set_goal_position(self.current_position + command)


def main(args=None):

    try:
        
        print("Force Control node is running...")

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