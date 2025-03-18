import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

import threading

import cv2
from cv_bridge import CvBridge
import ffmpegcv
from ffmpegcv.ffmpeg_reader_camera import query_camera_devices
import numpy as np


class GelSightSensorError(RuntimeError):
    pass


class GelSight:
    """
    Class to connect to the GelSight sensor using the ffmpegcv package.
    Credit to Janis Lenz, Alina Böhm, Inga Pfenning for the code.
    """

    def __init__(self, img_shape: tuple[int, int] = (480, 320)):
        """
        Initialize connection to the GelSight sensor using the ffmpegcv package.

        :param img_shape: shape of the image (default: (480, 320))
        :return: None
        """

        self.__camera_name: str = "GelSight"
        self.__img_shape = img_shape
        self.__latest_img = np.zeros(tuple(reversed(img_shape)) + (3,), dtype=np.uint8)
        self.__img_before_latest = np.zeros(
            tuple(reversed(img_shape)) + (3,), dtype=np.uint8
        )
        self.__camera = ffmpegcv.VideoCaptureCAM(
            self._get_camera_id(), pix_fmt="rgb24", resize=self.__img_shape
        )
        if not self.connected:
            pass

        self.terminate_thread = False
        self.lock = threading.Lock()
        self.thread = threading.Thread(
            target=self._image_thread, name=f"{self.__camera_name}_thread"
        )
        self.thread.start()


    def __del__(self):
        """
        Destructor for the GelSight sensor.

        :return: None
        """

        self.close()


    def close(self):
        """
        Close the connection to the GelSight.
        
        :return: None
        """
        
        self.terminate_thread = True
        self.thread.join()
        self.__camera.close()


    def _image_thread(self):
        """
        Thread function to receive the images asynchronously.

        :return: None
        """
        
        try:
            while not self.terminate_thread:
                success, img = self.__camera.read()
                if not success:
                    print(f"{self.__camera_name} thread closed!")
                    self.terminate_thread = True
                    continue
                with self.lock:
                    self.__latest_img = img
                    self.__img_before_latest = self.__latest_img

        except Exception as ex:
            pass


    def _get_camera_id(self):
        """
        Determine camera ID of the GelSight.

        :raises: GelSightSensorError: ff no camera could be found
        :return: int: ID of the camera
        """

        devices = query_camera_devices()
        camera_ids = list(devices.keys())

        for id in camera_ids:
            name = str(devices[id][0])
            if self.__camera_name.lower() in name.lower():  # (id != 0) and
                print(f"Found {self.__camera_name}: {devices[id]}, id: {id}")
                return id
            
        raise GelSightSensorError(f"Could not find {self.__camera_name}!")


    def __enter__(self):
        """
        Context manager to use the GelSight sensor.

        :return: self
        """

        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager to close the connection to the GelSight sensor.

        :param exc_type: exception type
        :param exc_val: exception value
        :param exc_tb: exception traceback
        :return: None
        """

        self.close()


    @property
    def camera_name(self):
        """
        Camera name.

        :return: string: camera name
        """

        return self.__camera_name


    @property
    def camera(self) -> ffmpegcv.FFmpegReaderCAM:
        """
        Camera object.

        :return: ffmpegcv.FFmpegReaderCAM: camera object
        """
        
        return self.__camera


    @property
    def connected(self):
        """
        Connected to the the GelSight.

        :return: bool: true if connected, otherwise false
        """
        
        return self.__camera is not None and self.__camera.isOpened()


    @property
    def img_size(self) -> tuple[int, int]:
        """
        Shape of the image.

        :return: tuple[int, int]: shape
        """

        return self.__img_shape


    @property
    def latest_img(self) -> cv2.typing.MatLike:
        """
        Latest captured image of the GelSight sensor.

        :raises: GelSightSensorError: if connection lost
        :return: cv2.typing.MatLike: latest image
        """

        if not self.connected and not self.thread.is_alive():
            raise GelSightSensorError(f"Connection to {self.__camera_name} lost!")
        
        return self.__latest_img


    @property
    def second_latest_img(self) -> cv2.typing.MatLike:
        """
        Second latest captured image of the GelSight sensor.

        :raises: GelSightSensorError: if connection lost
        :return: cv2.typing.MatLike: second latest image
        """

        if not self.connected and not self.thread.is_alive():
            raise GelSightSensorError(f"Connection to {self.__camera_name} lost!")
        
        return self.__img_before_latest


class GelSightMiniNode(Node):

    def __init__(self):
        """
        This node publishes the sensor images from the GelSight Mini camera to the topic "gelsight_mini_image".

        :return: None
        """

        super().__init__("gelsight_mini_node")

        # initialize camera
        self.cam= GelSight(img_shape=(895, 672)) # size suggested by janos to maintain aspect ratio

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

        # get latest image from camera
        img = cv2.cvtColor(self.cam.latest_img, cv2.COLOR_BGR2RGB)

        # crop and resize
        border_size_x, border_size_y = int(img.shape[0] * (1 / 7)), int(np.floor(img.shape[1] * (1 / 7)))  # remove 1/7th of border from each size
        img = img[border_size_x:img.shape[0] - border_size_x, border_size_y:img.shape[1] - border_size_y]
        img = img[:, :-1]  # remove last column to get a popular image resolution
        img = cv2.resize(img, (self.imgw, self.imgh))  # final resize for 3d

        # convert to ros msg image
        img_msg = self.bridge.cv2_to_imgmsg(img, "bgr8")
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