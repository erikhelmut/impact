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
        self.marker_publisher = []
        for m in range(2):
            self.marker_publisher.append(self.create_publisher(ArUcoMarkerStamped, f"aruco_marker_{m}", 10))

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
        
        midp1 = np.mean(corners[ids[0].item()][0], axis=0)
        midp2 = np.mean(corners[ids[1].item()][0], axis=0)
        dist = np.linalg.norm(midp1 - midp2)

        return dist, midp1, midp2


    def detect_aruco(self, msg):
        """
        TODO
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
        
        # TODO: check if it makes sense to hardcode the IDs to 0 and 1

        # check if ids has id 0 and 1
        if ids is not None and len(ids) == 2:
        
            # publish position of markers
            for id in range(2):
                aruco_msg = ArUcoMarkerStamped()
                aruco_msg.header = header
                aruco_msg.id = id
                aruco_msg.x = np.array([corners[id][0][0][0], corners[id][0][1][0], corners[id][0][2][0], corners[id][0][3][0]], dtype=np.uint8)
                aruco_msg.y = np.array([corners[id][0][0][1], corners[id][0][1][1], corners[id][0][2][1], corners[id][0][3][1]], dtype=np.uint8)
                self.marker_publisher[id].publish(aruco_msg)

            # publish distance between markers
            dist, midp1, midp2 = self.calculate_distance(corners, ids)
            dist_msg = ArUcoDistStamped()
            dist_msg.header = header
            dist_msg.ids = ids.flatten().astype(np.uint8)
            dist_msg.distance = dist.item()
            self.distance_publisher.publish(dist_msg)
        
        else:
            dist = None

        # draw the detected markers in the image and publish it
        if ids is not None:
                
            aruco_img = cv2.aruco.drawDetectedMarkers(color_img, corners, ids)

            # draw the line between two markers
            if dist is not None:
                aruco_img = cv2.line(color_img, (int(midp1[0]), int(midp1[1])), (int(midp2[0]), int(midp2[1])), (0, 255, 0), 2)
            
            # publish the image with detected markers
            image_msg = Image()
            image_msg.header = header
            image_msg.data = aruco_img.tobytes()
            image_msg.height = aruco_img.shape[0]
            image_msg.width = aruco_img.shape[1]
            image_msg.encoding = "bgr8"
            image_msg.step = aruco_img.shape[1] * 3
            self.image_publisher.publish(image_msg)


def main(args=None):
    """
    ROS node for detecting ArUco markers in the color image of the RealSense D405 camera.

    :return: None
    """

    try:

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