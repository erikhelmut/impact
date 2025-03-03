import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from impact_interfaces.msg import ArUcoMarkerStamped, ArUcoDistStamped

import numpy as np
import cv2


class DetectArUcoNode(Node):

    def __init__(self):
        """
        This node subscribes to the color image topic of the RealSense D405 camera and detects ArUco markers in the image.

        :return: None
        """

        super().__init__("detect_aruco_node")

        # load the ArUco dictionary
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        self.parameters = cv2.aruco.DetectorParameters()

        # create subscriber to detect ArUco markers
        self.subscriber = self.create_subscription(Image, "realsense_d405_color_image", self.detect_aruco, 10)
        self.subscriber  # prevent unused variable warning

        # create publishers for detected markers with id 0 and 1
        self.marker_0_publisher = self.create_publisher(ArUcoMarkerStamped, "aruco_marker_0", 10)
        self.marker_1_publisher = self.create_publisher(ArUcoMarkerStamped, "aruco_marker_1", 10)
        self.marker_publisher = {0: self.marker_0_publisher, 1: self.marker_1_publisher}

        # create publisher for distance between markers
        self.distance_publisher = self.create_publisher(ArUcoDistStamped, "aruco_distance", 10)

        # create publisher for image with detected markers and distance
        self.image_publisher = self.create_publisher(Image, "realsense_d405_color_image_aruco", 10)


    def calculate_distance(self, corners, ids):
        """
        Calculate the distance between two ArUco markers.

        :param corners: detected corners
        :param ids: IDs of two markers
        :return: distance between two markers and their midpoints
        """
        
        midp0 = np.mean(corners[ids[0].item()][0], axis=0)
        midp1 = np.mean(corners[ids[1].item()][0], axis=0)
        dist = np.linalg.norm(midp0 - midp1)

        return dist, midp0, midp1


    def detect_aruco(self, msg):
        """
        Callback function to detect ArUco markers in the color image.

        :param msg: color image message
        :return: None
        """

        # convert the image message to an OpenCV image
        color_img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)

        # convert the color image to grayscale
        gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

        # detect the markers
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)

        # extract header from image message for timestamp
        header = Header()
        header.stamp = msg.header.stamp

        # publish the position of the detected markers
        if ids is not None:
            if len(ids) == 2 and (0 in ids and 1 in ids):
                # if both markers are detected, publish the position of both markers
                for id in ids.flatten():
                    aruco_msg = ArUcoMarkerStamped()
                    aruco_msg.header = header
                    aruco_msg.id = int(id)
                    idx = int(np.where(ids.flatten() == id)[0].item())
                    aruco_msg.x = np.array([corners[idx][0][0][0], corners[idx][0][1][0], corners[idx][0][2][0], corners[idx][0][3][0]], dtype=np.uint16)
                    aruco_msg.y = np.array([corners[idx][0][0][1], corners[idx][0][1][1], corners[idx][0][2][1], corners[idx][0][3][1]], dtype=np.uint16)
                    self.marker_publisher[id].publish(aruco_msg)
            elif len(ids) == 1 and (0 in ids or 1 in ids):
                # if only one marker is detected, publish the position of the marker (corners is in this case a tuple with one element)
                    aruco_msg = ArUcoMarkerStamped()
                    aruco_msg.header = header
                    aruco_msg.id = int(ids[0].item())
                    aruco_msg.x = np.array([corners[0][0][0][0], corners[0][0][1][0], corners[0][0][2][0], corners[0][0][3][0]], dtype=np.uint16)
                    aruco_msg.y = np.array([corners[0][0][0][1], corners[0][0][1][1], corners[0][0][2][1], corners[0][0][3][1]], dtype=np.uint16)
                    self.marker_publisher[ids[0].item()].publish(aruco_msg)

        # draw the detected markers in the image and publish it
        if ids is not None:

            if 0 in ids and 1 in ids:
                # publish distance between markers
                dist, midp0, midp1 = self.calculate_distance(corners, ids)
                dist_msg = ArUcoDistStamped()
                dist_msg.header = header
                dist_msg.ids = ids.flatten().astype(np.uint8)
                dist_msg.distance = dist.item()
                self.distance_publisher.publish(dist_msg)
            else:
                dist = None
            
            # draw the detected markers
            color_img = cv2.aruco.drawDetectedMarkers(color_img, corners, ids)

            # draw the line between two markers
            if dist is not None:
                color_img = cv2.line(color_img, (int(midp0[0]), int(midp0[1])), (int(midp1[0]), int(midp1[1])), (0, 255, 0), 2)
            
        # publish the image with detected markers and distance (if available)
        image_msg = Image()
        image_msg.header = header
        image_msg.data = color_img.tobytes()
        image_msg.height = color_img.shape[0]
        image_msg.width = color_img.shape[1]
        image_msg.encoding = "bgr8"
        image_msg.step = color_img.shape[1] * 3
        self.image_publisher.publish(image_msg)


def main(args=None):
    """
    ROS node for detecting ArUco markers in the color image of the RealSense D405 camera.

    :return: None
    """

    try:

        print("""
        DETECT ARUCO
         ██████████
         █ ██  █  █
         █  ████ ██
         █ █  ██  █
         █ ██  ████
         ██████████
        """)

        print("Detect ArUco Node is running... Press <ctrl> <c> to stop. \nPosition of ArUco markers with id 0 and 1 are being published on topics /aruco_marker_0 and /aruco_marker_1. Distance between markers is being published on topic /aruco_distance. Image with detected markers and distance is being published on topic /realsense_d405_color_image_aruco. \n")

        rclpy.init(args=args)

        detect_aruco_node = DetectArUcoNode()

        rclpy.spin(detect_aruco_node)

    finally:

        # Destroy the node explicitly
        # (optional - otherwise it will be done automatically
        # when the garbage collector destroys the node object)
        detect_aruco_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":

    main()