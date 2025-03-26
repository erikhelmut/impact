import threading
import pyrealsense2 as rs

import numpy as np
import cv2


class RealSenseD405Error(RuntimeError):
    pass


class RealSenseD405:

    def __init__(self, img_shape: tuple[int, int] = (848, 480)):
        """
        Initialize connection to the RealSense D405 camera.

        :param img_shape: shape of the image (default: (848, 480))  
        :return: None
        """
        
        # defince camera properties
        self.__camera_name: str = "RealSenseD405"
        self.__img_shape = img_shape
        self.__latest_color_image = np.zeros(tuple(reversed(img_shape)) + (3,), dtype=np.uint8)
        self.__color_image_before_latest = np.zeros(
            tuple(reversed(img_shape)) + (3,), dtype=np.uint8
        )
        self.__latest_depth_image = np.zeros(tuple(reversed(img_shape)) + (1,), dtype=np.uint8)
        self.__depth_image_before_latest = np.zeros(
            tuple(reversed(img_shape)) + (1,), dtype=np.uint8
        )

        # configure depth and color streams
        self.__camera = rs.pipeline()
        self.config = rs.config()

        # get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.__camera)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        # enable depth and color streams
        self.config.enable_stream(rs.stream.depth, img_shape[0], img_shape[1], rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, img_shape[0], img_shape[1], rs.format.bgr8, 30)

        # check if connected
        if not self.connected:
            pass

        # start streaming
        self.__camera.start(self.config)
        self.terminate_thread = False
        self.lock = threading.Lock()
        self.thread = threading.Thread(
            target=self._image_thread, name=f"{self.__camera_name}_thread"
        )
        self.thread.start()


    def __del__(self):
        """
        Destructor for the RealSense D405 camera.

        :return: None
        """

        self.close()


    def close(self):
        """
        Close the connection to the RealSense D405 camera.

        :return: None
        """

        self.terminate_thread = True
        self.thread.join()
        self.__camera.stop()


    def _image_thread(self):
        """
        Thread function to receive the images asynchronously.

        :return: None
        """

        try:
            while not self.terminate_thread:
                
                try:
                    # wait for a coherent pair of frames: depth and color
                    frames = self.__camera.wait_for_frames(timeout_ms=1000)
                except RuntimeError as _:
                    # try to reconnect
                    self.__camera.stop()
                    self.__camera = rs.pipeline()
                    self.__camera.start(self.config)
                    try:
                        frames = self.__camera.wait_for_frames()
                    except RuntimeError as _:
                        print(f"{self.__camera_name} thread closed!")
                        self.terminate_thread = True
                        continue

                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()

                if not depth_frame or not color_frame:
                    print(f"{self.__camera_name} thread closed!")
                    self.terminate_thread = True
                    continue
                
                with self.lock:
                    # convert images to numpy arrays
                    depth_image = np.asanyarray(depth_frame.get_data())
                    color_image = np.asanyarray(color_frame.get_data())

                    # update images
                    self.__color_image_before_latest = self.__latest_color_image.copy()
                    self.__depth_image_before_latest = self.__latest_depth_image.copy()

                    self.__latest_color_image = color_image
                    self.__latest_depth_image = depth_image

        except RealSenseD405Error as rserr:
            raise rserr


    def __enter__(self):
        """
        Context manager entry to connect to the RealSense D405 camera.

        :return: self
        """

        return self


    def __exit__(self):
        """
        Context manager exit to close the connection to the RealSense D405 camera.

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
    def camera(self) -> rs.pipeline:
        """
        Camera object.

        :return: rs.pipeline: camera object
        """

        return self.__camera


    @property
    def connected(self) -> bool:
        """
        Connected to the RealSense D405.

        :return: bool: true if connected, otherwise false
        """
        
        return self.__camera is not None


    @property
    def img_size(self) -> tuple[int, int]:
        """
        Shape of the image.

        :return: tuple[int, int]: shape
        """

        return self.__img_shape
    

    @property
    def latest_img(self) -> tuple[cv2.typing.MatLike, cv2.typing.MatLike]:
        """
        Latest captured color and depth images of the RealSense D405 camera.

        :raises: RealSenseD405Error: if connection lost
        :return: tuple(cv2.typing.MatLike, cv2.typing.MatLike): latest color and depth images
        """
        
        if not self.connected and not self.thread.is_alive():
            raise RealSenseD405Error(f"Connection to {self.__camera_name} lost!")
        
        return self.__latest_color_image, self.__latest_depth_image


    @property
    def second_latest_img(self) -> tuple[cv2.typing.MatLike, cv2.typing.MatLike]:
        """
        Second latest captured color and depth images of the RealSense D405 camera.

        :raises: RealSenseD405Error: if connection lost
        :return: tuple(cv2.typing.MatLike, cv2.typing.MatLike): latest color and depth images
        """
        
        if not self.connected and not self.thread.is_alive():
            raise RealSenseD405Error(f"Connection to {self.__camera_name} lost!")
        
        return self.__color_image_before_latest, self.__depth_image_before_latest