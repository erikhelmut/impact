import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from std_msgs.msg import Header
from geometry_msgs.msg import TransformStamped

from scipy.spatial.transform import Rotation
import numpy as np

from optitrack.optitrack import OptiTrack


class OptitrackNode(Node):

    def __init__(self):
        """
        This node is responsible for publishing the current position and orientation of the end-effector. 
        
        :return: None
        """

        super().__init__("optitrack_node")

        # initialize the OptiTrack instance
        self.ot = OptiTrack()

        # get transformation matrix
        self.transformation = self.ot.transformation

        # declare parameters for QoS settings
        self.declare_parameter("ot.qos.reliability", "reliable")
        self.declare_parameter("ot.qos.history", "keep_last")
        self.declare_parameter("ot.qos.depth", 10)

        # get QoS profile
        qos_profile = self.get_qos_profile("ot.qos")

        # create publisher for end-effector position and orientation
        self.optitrack_publisher = self.create_publisher(TransformStamped, "optitrack_ee_state", qos_profile)
        timer_period = 0.02  # 50 Hz
        self.timer = self.create_timer(timer_period, self.get_current_state)

        # load and set the optitrack to panda end-effector calibration
        self.load_ee_calibration()

        # start the Panda instance to retrieve end-effector position and orientation
        self.show_panda = True
        if self.show_panda:
            self.start_panda_instance()


    def __del__(self):
        """
        Destructor to clean up resources.
        
        return: None
        """

        self.ot.close()

    
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
    
    def start_panda_instance(self):
        """
        Start the Panda instance to retrieve end-effector position and orientation.

        :return: None
        """

        from franka_panda.panda_real import PandaReal
        import yaml

        with open("/home/erik/impact/src/franka_panda/config/panda_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        self.panda = PandaReal(config)


    def load_ee_calibration(self):
        """
        Load the calibration from optitrack rigid body to panda end-effector.

        :return: None
        """

        self.ee_offset = np.load("/home/erik/impact/src/optitrack/calibration/ee_offset.npy")  # old: translate_ot_panda
        self.ee_rotation = np.load("/home/erik/impact/src/optitrack/calibration/ee_rotation.npy")  # old: rot_ori_ot_panda


    def get_current_state(self):
        """
        Retrieve the current state of the end-effector and publish it.

        :return: None
        """

        # extract panda end-effector and OptiTrack rigid body positions and orientations
        ot_rb_pos = self.ot.ee_pos
        ot_rb_ori = self.ot.ee_ori

        # extract rotation from transformation matrix
        R_T = self.transformation[:3, :3]

        # convert to quaternion
        q_T = Rotation.from_matrix(R_T).as_quat()

        # rotation matrix from OptiTrack Rigid Body in optitrack coordinates
        R_OT_RB = Rotation.from_quat(ot_rb_ori)
        
        # rotation matrix from OpiTrack Rigid Body in panda coordinates
        R_T = Rotation.from_quat(q_T)
        R_OT_PANDA_RB = (R_T * R_OT_RB).as_quat()

        # transform optitrack rigid body position to panda coordinates
        ot_panda_ee_pos = self.transformation[:3, :3] @ ot_rb_pos + self.transformation[:3, 3]


        # publish the optitrack rigid body position and orientation
        msg = TransformStamped()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "panda_base"
        msg.child_frame_id = "ot_ee"
        msg.transform.translation.x = ot_panda_ee_pos[0]
        msg.transform.translation.y = ot_panda_ee_pos[1]
        msg.transform.translation.z = ot_panda_ee_pos[2]
        msg.transform.rotation.x = R_OT_PANDA_RB[0]
        msg.transform.rotation.y = R_OT_PANDA_RB[1]
        msg.transform.rotation.z = R_OT_PANDA_RB[2]
        msg.transform.rotation.w = R_OT_PANDA_RB[3]

        self.optitrack_publisher.publish(msg)


        # calculate the optitrack end-effector position and orientation of the panda end-effector
        panda_ot_ee_pos_calculated = ot_panda_ee_pos + self.ee_offset
        panda_ot_ee_ori_calculated = Rotation.from_quat(R_OT_PANDA_RB) * Rotation.from_quat(self.ee_rotation)
        panda_ot_ee_ori_calculated = panda_ot_ee_ori_calculated.as_quat()

        # publish the optritrack end-effector position and orientation of the panda end-effector
        msg = TransformStamped()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "panda_base"
        msg.child_frame_id = "panda_ot_ee"
        msg.transform.translation.x = panda_ot_ee_pos_calculated[0]
        msg.transform.translation.y = panda_ot_ee_pos_calculated[1]
        msg.transform.translation.z = panda_ot_ee_pos_calculated[2]
        msg.transform.rotation.x = panda_ot_ee_ori_calculated[0]
        msg.transform.rotation.y = panda_ot_ee_ori_calculated[1]
        msg.transform.rotation.z = panda_ot_ee_ori_calculated[2]
        msg.transform.rotation.w = panda_ot_ee_ori_calculated[3]

        self.optitrack_publisher.publish(msg)


        if self.show_panda:
            # retrieve the end-effector position and orientation from the panda instance
            panda_ee_pos = self.panda.end_effector_position
            panda_ee_ori = self.panda.end_effector_orientation
            
            # publish the end-effector position and orientation of the panda instance
            msg = TransformStamped()
            msg.header = Header()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "panda_base"
            msg.child_frame_id = "panda_ee"
            msg.transform.translation.x = panda_ee_pos[0]
            msg.transform.translation.y = panda_ee_pos[1]
            msg.transform.translation.z = panda_ee_pos[2]
            msg.transform.rotation.x = panda_ee_ori[0]
            msg.transform.rotation.y = panda_ee_ori[1]
            msg.transform.rotation.z = panda_ee_ori[2]
            msg.transform.rotation.w = panda_ee_ori[3]
            
            self.optitrack_publisher.publish(msg)


def main(args=None):
    """
    ROS node for the OptiTrack system.

    :param args: arguments for the ROS node
    :return: None
    """

    try:

        print("Optitrack ROS node is running... Press <ctrl> <c> to stop. \nEndeffector state is being published on topic /optitrack_ee_state. \n")

        rclpy.init(args=args)

        optitrack_node = OptitrackNode()

        rclpy.spin(optitrack_node)

    finally:

        # Destroy the node explicitly
        # (optional - otherwise it will be done automatically
        # when the garbage collector destroys the node object)
        optitrack_node.__del__()
        optitrack_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    
    main()