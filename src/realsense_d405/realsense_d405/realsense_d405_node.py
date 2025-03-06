import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import Image

import pyrealsense2 as rs

import numpy as np


class RealSenseD405Node(Node):

    def __init__(self):
        """
        This node reads the RealSense D405 camera. It publishes the current color and depth frames and subscribes to a topic to calculate the position of the ArUco markers.

        :return: None
        """
        
        super().__init__("realsense_d405_node")

        # configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()

        # get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        # enable depth and color streams
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)

        # start streaming
        self.pipeline.start(config)

        # create publisher for color and depth image
        self.color_publisher_ = self.create_publisher(Image, "realsense_d405_color_image", 10)
        self.depth_publisher_ = self.create_publisher(Image, "realsense_d405_depth_image", 10)
        timer_period = 1.0 / 30  # 30 Hz
        self.timer = self.create_timer(timer_period, self.get_frames)


    def __del__(self):
        """
        Descructor for the RealSense D405 node.

        :return: None
        """
        
        self.pipeline.stop()


    def get_frames(self):
        """
        Callback function to publish the current color and depth frames.

        :return: None
        """

        # wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

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