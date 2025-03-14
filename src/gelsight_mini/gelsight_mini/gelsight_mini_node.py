import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import Image

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
        self.cam = cv2.VideoCapture(0)

        # set camera parameters
        self.imgw = 320
        self.imgh = 240

        # initialize cv bridge
        self.bridge = CvBridge()

        # create publisher for the GelSight Mini
        self.gs_mini_publisher_ = self.create_publisher(Image, "gelsight_mini_image", 10)
        timer_period = 1.0 / 25  # 25 Hz
        self.timer = self.create_timer(timer_period, self.get_image)


    def get_image(self):
        """
        Callback function to publish the current image from the GelSight Mini camera.

        :return: None
        """

        # read image from camera
        ret, f0 = self.cam.read()

        if ret:
            # resize, crop and resize back
            img = cv2.resize(f0, (895, 672))  # size suggested by janos to maintain aspect ratio
            border_size_x, border_size_y = int(img.shape[0] * (1 / 7)), int(np.floor(img.shape[1] * (1 / 7)))  # remove 1/7th of border from each size
            img = img[border_size_x:img.shape[0] - border_size_x, border_size_y:img.shape[1] - border_size_y]
            img = img[:, :-1]  # remove last column to get a popular image resolution
            img = cv2.resize(img, (self.imgw, self.imgh))  # final resize for 3d

            # convert to ros msg image
            img_msg = self.bridge.cv2_to_imgmsg(img, "bgr8")
            img_msg.header.stamp = self.get_clock().now().to_msg()
            self.gs_mini_publisher_.publish(img_msg)

        else:
            print("ERROR! reading image from camera")

        
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

        print("GelSight Mini Node is running.. Press <ctrl> <c> to stop. \nGelSight Mini images are being published on topic /gelsight_mini_image. \n")

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