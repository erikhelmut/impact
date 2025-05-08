import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import Header
from sensor_msgs.msg import Image

from realsense_d405.rsdevice import RealSenseD405


class RealSenseD405Node(Node):

    def __init__(self):
        """
        This node reads the RealSense D405 camera. It publishes the current color and depth frames and subscribes to a topic to calculate the position of the ArUco markers.

        :return: None
        """
        
        super().__init__("realsense_d405_node")

        # initialze camera
        self.cam = RealSenseD405(img_shape=(848, 480))
        #self.cam = RealSenseD405(img_shape=(640, 480))

        # declare parameters for QoS settings
        self.declare_parameter("rs_d405.qos.reliability", "reliable")
        self.declare_parameter("rs_d405.qos.history", "keep_last")
        self.declare_parameter("rs_d405.qos.depth", 10)

        # get QoS profile
        qos_profile = self.get_qos_profile("rs_d405.qos")

        # create publisher for color and depth image
        self.color_publisher_ = self.create_publisher(Image, "realsense_d405_color_image", qos_profile)
        self.depth_publisher_ = self.create_publisher(Image, "realsense_d405_depth_image", qos_profile)
        timer_period = 1.0 / 30  # 30 Hz
        self.timer = self.create_timer(timer_period, self.get_frames)


    def __del__(self):
        """
        Descructor for the RealSense D405 node.

        :return: None
        """
        
        self.cam.__del__()


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


    def get_frames(self):
        """
        Callback function to publish the current color and depth frames.

        :return: None
        """

        # convert images to numpy arrays
        color_image, depth_image = self.cam.latest_img

        header = Header()
        header.stamp = self.get_clock().now().to_msg()

        # create Image messages
        color_msg = Image()
        color_msg.header = header
        color_msg.data = color_image.tobytes()
        color_msg.height = color_image.shape[0]
        color_msg.width = color_image.shape[1]
        color_msg.encoding = "bgr8"
        color_msg.step = color_image.shape[1] * 3

        depth_msg = Image()
        depth_msg.header = header
        depth_msg.data = depth_image.tobytes()
        depth_msg.height = depth_image.shape[0]
        depth_msg.width = depth_image.shape[1]
        depth_msg.encoding = "16UC1"
        depth_msg.step = depth_image.shape[1] * 2

        # publish the images
        self.color_publisher_.publish(color_msg)
        self.depth_publisher_.publish(depth_msg)


def main(args=None):
    """
    ROS node for the Intel RealSense D405 camera.
    
    :param args: arguments for the ROS node
    :return: None
    """

    try:

        print("""
         INTEL REALSENSE D405  
        .===================.
        |                   |
        |                   |    
        |   [ O ]   [ O ]   |
        |                   |
        |                   |
        '==================='  
        """)

        print("RealSense D405 Node is running... Press <ctrl> <c> to stop. \nColor and depth images are being published on topics /realsense_d405_color_image and /realsense_d405_depth_image. \n")

        rclpy.init(args=args)

        realsense_d405_node = RealSenseD405Node()

        rclpy.spin(realsense_d405_node)

    finally:

        # Destroy the node explicitly
        # (optional - otherwise it will be done automatically
        # when the garbage collector destroys the node object)
        realsense_d405_node.__del__()
        realsense_d405_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":

    main()