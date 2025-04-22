import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image

from gelsight_mini.gsdevice import GelSight

import cv2
from cv_bridge import CvBridge

import numpy as np


class GelSightMiniNode(Node):

    def __init__(self):
        """
        This node publishes the sensor images from the GelSight Mini camera to the topic "gelsight_mini_image".

        :return: None
        """

        super().__init__("gelsight_mini_node")

        # initialize camera
        self.cam = GelSight(img_shape=(895, 672)) # size suggested by janos to maintain aspect ratio

        # set camera parameters
        self.imgw = 320
        self.imgh = 240

        # initialize cv bridge
        self.bridge = CvBridge()
        
        # declare parameters for QoS settings
        self.declare_parameter("gs_mini.qos.reliability", "reliable")
        self.declare_parameter("gs_mini.qos.history", "keep_last")
        self.declare_parameter("gs_mini.qos.depth", 10)

        # get QoS profile
        qos_profile = self.get_qos_profile("gs_mini.qos")
        
        # create publisher for the GelSight Mini
        self.gs_mini_publisher_ = self.create_publisher(Image, "gelsight_mini_image", qos_profile)
        timer_period = 1.0 / 25  # 25 Hz
        self.timer = self.create_timer(timer_period, self.get_image)

    
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


    def get_image(self):
        """
        Callback function to publish the current image from the GelSight Mini camera.

        :return: None
        """

        # get latest image from camera
        img = self.cam.latest_img

        # crop and resize
        border_size_x, border_size_y = int(img.shape[0] * (1 / 7)), int(np.floor(img.shape[1] * (1 / 7)))  # remove 1/7th of border from each size
        img = img[border_size_x:img.shape[0] - border_size_x, border_size_y:img.shape[1] - border_size_y]
        img = img[:, :-1]  # remove last column to get a popular image resolution
        img = cv2.resize(img, (self.imgw, self.imgh))  # final resize for 3d

        # convert to ros image message
        img_msg = self.bridge.cv2_to_imgmsg(img, "rgb8")
        img_msg.header.stamp = self.get_clock().now().to_msg()
        self.gs_mini_publisher_.publish(img_msg)

        
def main(args=None):
    """
    ROS node for the GelSight Mini sensor.

    :param args: arguments for the ROS node
    :return: None
    """

    try:

        print(""""
           GELSIGHT MINI
        .------------------.
        |.----------------.|
        |  ______________  |
        | |              | |
        | |              | |
        | |______________| |
        |__________________|
        """)

        print("GelSight Mini Node is running... Press <ctrl> <c> to stop. \nGelSight Mini images are being published on topic /gelsight_mini_image. \n")

        rclpy.init(args=args)

        gelsight_mini_node = GelSightMiniNode()

        rclpy.spin(gelsight_mini_node)

    finally:

        # Destroy the node explicitly
        # (optional - otherwise it will be done automatically
        # when the garbage collector destroys the node object)
        gelsight_mini_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    
    main()