import rclpy
from rclpy.node import Node
from message_filters import Subscriber, ApproximateTimeSynchronizer
from std_msgs.msg import Int16, Header
from sensor_msgs.msg import JointState
from impact_interfaces.msg import ArUcoDistStamped

import numpy as np
import time
import matplotlib.pyplot as plt


class CalibrateGripper(Node):

    def __init__(self):

        super().__init__("calibrate_gripper")

        # create subscriber for aruco distance
        self.subscriber_aruco_dist = Subscriber(self, ArUcoDistStamped, "aruco_distance")

        # create subscriber for motor state
        self.subscriber_motor_state = Subscriber(self, JointState, "actuated_umi_motor_state")

        # create publisher to set goal position of gripper
        self.publisher = self.create_publisher(Int16, "set_actuated_umi_motor_position", 10)

        # set start position of gripper
        self.set_goal_position(900)

        # save msgs
        self.aruco_dist = []
        self.motor_pos = []

        # create synchronizer
        self.sync = ApproximateTimeSynchronizer(
            [self.subscriber_aruco_dist, self.subscriber_motor_state], queue_size=10, slop=0.1  # 100 ms
        )
        self.sync.registerCallback(self.sync_callback)


    def sync_callback(self, msg_aruco_dist, msg_motor_state):

        self.aruco_dist.append(msg_aruco_dist.distance)
        self.motor_pos.append(msg_motor_state.position[0])


    def set_goal_position(self, position):

        msg = Int16()
        msg.data = int(position)
        self.publisher.publish(msg)


def main(args=None):

    try:
        
        rclpy.init(args=args)

        calibrate_gripper = CalibrateGripper()

        goal_position = 1000
        while rclpy.ok():
            rclpy.spin_once(calibrate_gripper, timeout_sec=1.0)

            # call every 1 seconds set_goal_position
            current_time = time.time()
            if not hasattr(calibrate_gripper, 'last_call_time'):
                calibrate_gripper.last_call_time = current_time

            if current_time - calibrate_gripper.last_call_time >= 0.1:
                calibrate_gripper.set_goal_position(goal_position)  # Example position
                goal_position -= 10
                calibrate_gripper.last_call_time = current_time

            if goal_position < -2380:
                break
    

        # plot motor position vs aruco distance
        plt.scatter(calibrate_gripper.motor_pos, calibrate_gripper.aruco_dist)

        # make least squares fit
        A = np.vstack([calibrate_gripper.motor_pos, np.ones(len(calibrate_gripper.motor_pos))]).T
        m, c = np.linalg.lstsq(A, calibrate_gripper.aruco_dist[50:], rcond=None)[0]
        plt.plot(calibrate_gripper.motor_pos, m*np.array(calibrate_gripper.motor_pos) + c, 'r')

        plt.xlabel("Motor Position")
        plt.ylabel("ArUco Distance")
        plt.title("Motor Position vs ArUco Distance")
        plt.show()

    finally:
        
        # Destroy the node explicitly
        # (optional - otherwise it will be done automatically
        # when the garbage collector destroys the node object)
        calibrate_gripper.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    
    main()