import time
import numpy as np
from scipy.spatial.transform import Rotation
from transformation import Transformation

from natnet_client import DataDescriptions, DataFrame, NatNetClient


SERVER_IP_ADDRESS = "10.90.90.5"
LOCAL_IP_ADDRESS = "10.90.90.21"

CYLINDER_NAME = "dreamer_cylinder"
BASE_NAME = "dreamer_base"
END_EFFECTOR = "dreamer_ee"


class OptiTrack:
    # constants
    BASE_TO_HOLE = np.array([0.0061, -0.035, 0.0274])
    BASE_TO_RESET = np.array([-0.111, 0.0, -0.034])

    def __init__(self):
        self._transformation = None
        try:
            self._transformation = np.load("calibration.npy")
            print("Calibration loaded.")
        except:
            print("No saved calibration found!")

        # id's
        self.cylinder_id: int = None
        self.base_id: int = None
        self.ee_id: int = None

        # raw OptiTrack data
        self._base_pos = np.zeros(3)
        self._base_ori = np.zeros(4)
        self._cyl_pos = np.zeros(3)
        self._cyl_ori = np.zeros(4)
        self._ee_pos = np.zeros(3)
        self._ee_ori = np.zeros(4)

        # calculated positions
        self._goal_pos = np.zeros(3)
        self._reset_pos = np.zeros(3)

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
        if self.cylinder_id is None or self.base_id is None:
            return

        for rb in data_frame.rigid_bodies:
            if rb.id_num == self.base_id:
                self._base_pos = np.array(rb.pos)
                self._base_ori = np.array(rb.rot)
            elif rb.id_num == self.cylinder_id:
                self._cyl_pos = np.array(rb.pos)
                self._cyl_ori = np.array(rb.rot)
            elif rb.id_num == self.ee_id:
                self._ee_pos = np.array(rb.pos)
                self._ee_ori = np.array(rb.rot)

        rot = Rotation.from_quat(self._base_ori)
        rot = rot.as_matrix()
        self._goal_pos = self._base_pos + (rot @ self.BASE_TO_HOLE)
        self._reset_pos = self._base_pos + (rot @ self.BASE_TO_RESET)
        self._last_update_time = time.time()

    def receive_new_desc(self, desc: DataDescriptions):
        print("New desc!")
        for rb in desc.rigid_bodies:
            if rb.name == BASE_NAME:
                self.base_id = rb.id_num
                print(f'The base called "{rb.name}" has ID {rb.id_num}')
            if rb.name == CYLINDER_NAME:
                self.cylinder_id = rb.id_num
                print(f'The cylinder called "{rb.name}" has ID {rb.id_num}')
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
    def base_pos(self) -> np.ndarray:
        return self._base_pos

    @property
    def base_ori(self) -> np.ndarray:
        return self._base_ori

    @property
    def cylinder_pos(self) -> np.ndarray:
        return self._cyl_pos  # + np.array([0, 0, 0.002])

    @property
    def cylinder_pos_bottom(self) -> np.ndarray:
        rot = Rotation.from_quat(self.cylinder_ori)
        return self._cyl_pos - np.array(rot.apply(np.array([0, 0.088, 0])))

    @property
    def cylinder_pos_gripper(self) -> np.ndarray:
        rot = Rotation.from_quat(self.cylinder_ori)
        return self._cyl_pos - np.array(rot.apply(np.array([0, 0.022, 0])))

    @property
    def cylinder_ori(self) -> np.ndarray:
        return self._cyl_ori

    @property
    def reset_pos(self) -> np.ndarray:
        return self._reset_pos

    @property
    def goal_pos(self) -> np.ndarray:
        return self._goal_pos

    @property
    def last_update_time(self) -> float:
        return self._last_update_time


if __name__ == "__main__":
    # This test is good to check the rotation axis
    ot = OptiTrack()
    time.sleep(2)
    try:
        while True:
            rot = Rotation.from_quat(ot.cylinder_ori)
            rot = rot.as_euler("zyx", degrees=True)
            print(f"Cylinder position: {ot.cylinder_pos}")
            print(f"Cylinder position bottom: {ot.cylinder_pos_bottom}")
            print(f"Cylinder orientation: {rot}")
            print(f"Time delta to last update: {time.time() - ot.last_update_time}\n")
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")
    ot.close()
