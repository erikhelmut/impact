import threading

import ffmpegcv
from ffmpegcv.ffmpeg_reader_camera import query_camera_devices

import numpy as np
import cv2


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
                    self.__img_before_latest = self.__latest_img.copy()
                    self.__latest_img = img

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
        Connected to the GelSight.

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