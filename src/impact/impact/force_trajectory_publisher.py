import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import WrenchStamped
from impact_msgs.msg import GoalForceController
from collections import deque


class ForceTrajectoryPublisher(Node):

    def __init__(self):
        """
        This node publishes the goal force of the gripper in a simple linear trajectory.

        :return: None
        """

        super().__init__("force_trajectory_publisher")

        # store control parameters
        self.kp = 0.0
        self.kd = 0.0
        self.alpha = 0.0

        # store current and goal force
        self.current_force = None
        self.goal_force = None
        self.internal_goal_force = None

        # create subscriber to get current force of gripper without callback
        self.current_force_subscriber = self.create_subscription(WrenchStamped, "resense_0", self.receive_force, 10)
        self.current_force_subscriber  # prevent unused variable warning
        # timer to process messages at 25 Hz (40 ms interval)
        self.resense_timer = self.create_timer(1.0 / 25, self.get_current_force)
        # buffer to store incoming messages
        self.message_queue = deque(maxlen=1)

        # create subscriber to set goal force of gripper
        self.goal_force_subscriber = self.create_subscription(GoalForceController, "set_resense_goal_force", self.set_force_goal, 10)

        # create publsisher to set goal force of gripper
        self.goal_force_publisher = self.create_publisher(GoalForceController, "set_actuated_umi_goal_force", 10)

        # create timer to control the force of the gripper
        self.control_frequency = 25
        self.force_goal_timer = self.create_timer(1.0 / self.control_frequency, self.pub_force_goal)


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

            # store current force
            self.current_force = msg.wrench.force.z


    def set_force_goal(self, msg):
        """
        Set the goal force of the gripper.

        :param msg: float message containing the goal normal force
        :return: None
        """
        
        self.goal_force = msg.goal_force
        self.internal_goal_force = self.current_force

        self.kp = msg.kp
        self.kd = msg.kd
        self.alpha = msg.alpha


    def pub_force_goal(self):
        """
        Publish the goal force of the gripper in a simple linear trajectory.

        :return: None
        """
        
        # check if goal force is set
        if self.goal_force is not None and self.internal_goal_force is not None:
            
            # check if internal goal force is 0.2 N away from goal force
            if abs(self.internal_goal_force - self.goal_force) > 0.2:

                # check if current force is smaller or bigger than goal force
                if self.current_force > self.goal_force:
                    # update internal goal force
                    self.internal_goal_force -= 0.02
                elif self.current_force < self.goal_force:
                    # update internal goal force
                    self.internal_goal_force += 0.02
            else:
                # set goal force to goal force
                self.internal_goal_force = self.goal_force

            # publish goal force
            msg = GoalForceController()
            msg.goal_force = self.internal_goal_force
            msg.kp = self.kp
            msg.kd = self.kd
            msg.alpha = self.alpha
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