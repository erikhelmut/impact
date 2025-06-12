import time
import numpy as np
from scipy.spatial.transform import Rotation
from transformation import Transformation

from natnet_client import DataDescriptions, DataFrame, NatNetClient


SERVER_IP_ADDRESS = "10.90.90.5"
LOCAL_IP_ADDRESS = "10.90.90.21"

END_EFFECTOR = "erik_ee"


class OptiTrack:

    def __init__(self):
        self._transformation = None
        try:
            self._transformation = np.load("/home/erik/impact/src/optitrack/calibration/tf_calibration.npy")
            print("Calibration loaded.")
        except:
            print("No saved calibration found!")

        # id's
        self.ee_id: int = None

        # raw OptiTrack data
        self._ee_pos = np.zeros(3)
        self._ee_ori = np.zeros(4)

        # init streaming client
        self.streaming_client = NatNetClient(
            server_ip_address=SERVER_IP_ADDRESS,
            local_ip_address=LOCAL_IP_ADDRESS,
            multicast_address="239.255.42.99",
            use_multicast=True,
        )
        self.streaming_client.on_data_description_received_event.handlers.append(
            self.receive_new_desc
        )
        self.streaming_client.on_data_frame_received_event.handlers.append(
            self.receive_new_frame
        )
        self._last_update_time = None
        time.sleep(1)
        self.streaming_client.connect(timeout=10)
        self.streaming_client.run_async()
        self.streaming_client.request_modeldef()
        time.sleep(1)

    def close(self):
        self.streaming_client.shutdown()
        print("OptiTrack closed.")

    def receive_new_frame(self, data_frame: DataFrame):

        for rb in data_frame.rigid_bodies:
            if rb.id_num == self.ee_id:
                self._ee_pos = np.array(rb.pos)
                self._ee_ori = np.array(rb.rot)

        self._last_update_time = time.time()

    def receive_new_desc(self, desc: DataDescriptions):
        print("New desc!")
        for rb in desc.rigid_bodies:
            if rb.name == END_EFFECTOR:
                self.ee_id = rb.id_num
                print(f'The end-effector called "{rb.name}" has ID {rb.id_num}')
        print()

    def optitrack_to_panda(self, x, rotate_only=False):
        assert (
            self.transformation is not None
        ), "Cannot perform transformation without transformation matrix!"
        if rotate_only:
            T = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)
            return self.transformation[:3, :3] @ x
        else:
            x_extended = np.append(x, 1)
            return (self.transformation @ x_extended)[0:3]

    """
    Properties:
    - all coordinates in OptiTrack system (see optitrack_to_panda function)
    - rotations in quaternions
    """

    @property
    def transformation(self) -> np.ndarray:
        return self._transformation

    @transformation.setter
    def transformation(self, tr):
        if isinstance(tr, Transformation):
            self._transformation = tr.matrix
        elif isinstance(tr, np.ndarray):
            assert tr.shape == (4, 4), "Passed matrix isn't (4x4)!"
            self._transformation = tr

    @property
    def ee_pos(self) -> np.ndarray:
        return self._ee_pos

    @property
    def ee_ori(self) -> np.ndarray:
        return self._ee_ori

    @property
    def last_update_time(self) -> float:
        return self._last_update_time


if __name__ == "__main__":
    # This test is good to check the rotation axis
    ot = OptiTrack()
    time.sleep(2)
    try:
        while True:
            rot = Rotation.from_quat(ot.ee_ori)
            rot = rot.as_euler("zyx", degrees=True)
            print(f"End-effector position: {ot.ee_pos}")
            print(f"End-effector orientation: {rot}")
            print(f"Time delta to last update: {time.time() - ot.last_update_time}\n")
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")
    ot.close()
