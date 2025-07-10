import math
import sys
from enum import Enum
import yaml
import time
import numpy as np
import random
from scipy.spatial.transform import Rotation

from franky import (
    Gripper,
    Robot,
    Affine,
    CartesianMotion,
    ReferenceType,
    JointWaypoint,
    JointWaypointMotion,
    Reaction,
    Measure,
    Motion,
)
import franky


ROBOT_HOST = "10.10.10.10"
NEUTRAL_POS = [
    -1.49446089e-02,
    -6.01734484e-02,
    1.93335545e-03,
    -2.19032964e00,
    -2.28822809e-02,
    2.15411278e00,
    8.07744573e-01,
]
NEUTRAL_ORI = [9.99941792e-01, -5.83794716e-03, 1.03312910e-04, 9.07305035e-03]


class Axis(Enum):
    X = 0
    Y = 1
    Z = 2


class PandaReal:

    def __init__(self, config: dict):
        print("Initializing Robot...")

        # init robot
        self.robot = Robot(
            ROBOT_HOST, default_torque_threshold=500.0, default_force_threshold=500.0
        )
        #self.robot.set_joint_impedance(
        #    np.array([3.0, 3.0, 3.0, 2.5, 2.5, 2.0, 2.0]) * 100
        #)
        self.robot.set_joint_impedance(
            np.array([3.0, 3.0, 3.0, 2.5, 2.5, 2.0, 2.0]) * 500
        )
        inertia = np.eye(3) * 1e-4
        inertia = inertia.T.reshape(-1)
        self.robot.set_load(config["load"], config["center_mass"], inertia.tolist())
        self.robot.recover_from_errors()
        self.robot.relative_dynamics_factor = 0.05
        self._dflt_limit_forces = config["dflt_limit_forces"]
        self._max_forces = np.array(config["dflt_max_forces"])

        # init gripper
        #self.gripper = Gripper(ROBOT_HOST)
        #self._dflt_gripper_speed: float = config["dflt_gripper_speed"]
        #self._dflt_gripper_force: float = config["dflt_gripper_force"]

    # ==========================#
    # Properties               #
    # ==========================#
    @property
    def end_effector_position(self) -> np.ndarray:
        return self.robot.current_pose.end_effector_pose.translation

    @property
    def end_effector_orientation(self) -> np.ndarray:
        return self.robot.current_pose.end_effector_pose.quaternion

    @property
    def current_joint_positions(self) -> np.ndarray:
        return self.robot.current_joint_positions

    @property
    def max_forces(self) -> np.array:
        return self._max_forces

    @max_forces.setter
    def max_forces(self, forces: np.array) -> None:
        self._max_forces = forces

    @property
    def gripper_width(self) -> float:
        return self.gripper.width

    # ==========================#
    # Move functions           #
    # ==========================#
    def move_to_neutral(
        self, rel_vel: float, limit_forces: bool = None, asynch: bool = False
    ):
        motion = JointWaypointMotion([JointWaypoint(NEUTRAL_POS)])
        if (limit_forces is not None and limit_forces) or (
            limit_forces is None and self._dflt_limit_forces
        ):
            motion = self._add_max_forces(motion)
        self._exec_motion(motion, rel_vel, asynch)

    def move_to_joint_position(
        self,
        joint_positions: list,
        rel_vel: float,
        limit_forces: bool = None,
        asynch: bool = False,
    ):
        assert len(joint_positions) == 7, "There have to be 7 joint positions!"
        motion = JointWaypointMotion([JointWaypoint(joint_positions)])
        if (limit_forces is not None and limit_forces) or (
            limit_forces is None and self._dflt_limit_forces
        ):
            motion = self._add_max_forces(motion)
        self._exec_motion(motion, rel_vel, asynch)

    def move_in_axis_rel(
        self,
        axis: Axis,
        dist: float,
        rel_vel: float,
        goal_ori: np.ndarray = np.array(NEUTRAL_ORI),
        limit_forces: bool = None,
        asynch: bool = False,
    ):
        goal_pos = np.array(self.end_effector_position)
        goal_pos[axis.value] += dist
        motion = CartesianMotion(Affine(goal_pos, goal_ori), ReferenceType.Absolute)
        if (limit_forces is not None and limit_forces) or (
            limit_forces is None and self._dflt_limit_forces
        ):
            motion = self._add_max_forces(motion)
        self._exec_motion(motion, rel_vel, asynch)

    def move_in_axis_abs(
        self,
        axis: Axis,
        value: float,
        rel_vel: float,
        goal_ori: np.ndarray = np.array(NEUTRAL_ORI),
        limit_forces: bool = None,
        asynch: bool = False,
    ):
        goal_pos = np.array(self.end_effector_position)
        goal_pos[axis.value] = value
        motion = CartesianMotion(Affine(goal_pos, goal_ori), ReferenceType.Absolute)
        if (limit_forces is not None and limit_forces) or (
            limit_forces is None and self._dflt_limit_forces
        ):
            motion = self._add_max_forces(motion)
        self._exec_motion(motion, rel_vel, asynch)

    def move_rel(
        self,
        direction: np.ndarray,
        rel_vel: float,
        goal_ori: np.ndarray = np.array(NEUTRAL_ORI),
        limit_forces: bool = None,
        asynch: bool = False,
    ):
        goal_pos = np.array(self.end_effector_position)
        goal_pos += direction
        motion = CartesianMotion(Affine(goal_pos, goal_ori), ReferenceType.Absolute)
        if (limit_forces is not None and limit_forces) or (
            limit_forces is None and self._dflt_limit_forces
        ):
            motion = self._add_max_forces(motion)
        self._exec_motion(motion, rel_vel, asynch)

    def move_abs(
        self,
        goal_pos: np.ndarray,
        rel_vel: float,
        goal_ori: np.ndarray = np.array(NEUTRAL_ORI),
        limit_forces: bool = None,
        asynch: bool = False,
    ):
        motion = CartesianMotion(Affine(goal_pos, goal_ori), ReferenceType.Absolute)
        if (limit_forces is not None and limit_forces) or (
            limit_forces is None and self._dflt_limit_forces
        ):
            motion = self._add_max_forces(motion)
        self._exec_motion(motion, rel_vel, asynch)

    def _add_max_forces(self, motion: Motion):
        # no reaction movement when exeeding the threshholds
        reaction_motion = CartesianMotion(Affine([0.0, 0.0, 0.0]), ReferenceType.Relative)

        forces = [Measure.FORCE_X, Measure.FORCE_Y, Measure.FORCE_Z]
        for i in range(3):
            if self.max_forces[i] is not None:
                reaction = Reaction(forces[i] > self.max_forces[i], reaction_motion)
                motion.add_reaction(reaction)
                reaction.register_callback(
                    lambda *args: print(
                        f"Reaction triggered in positive {i} direction (0:X, 1:Y, 2:Z)"
                    )
                )
                reaction = Reaction(forces[i] < -self.max_forces[i], reaction_motion)
                motion.add_reaction(reaction)
                reaction.register_callback(
                    lambda *args: print(
                        f"Reaction triggered in negative {i} direction (0:X, 1:Y, 2:Z)"
                    )
                )

        return motion

    def _exec_motion(self, motion: JointWaypointMotion, rel_vel: float, asynch: bool):
        """
        The motions_only function moves the robot to a given motion, and then
        moves it down until the end-effector is in contact with an object. The
        motion is specified by a dictionary containing x, y, z coordinates for
        translation and rotation about each axis. If orientation_noisy=True (default),
        the function will add random noise to the rotation of the end-effector before moving it down.

        :param motion: Move the robot to a specific location
        :param orientation_noisy: Decide whether the robot should turn during its motion
        :return: The turning value
        :doc-author: Trelent
        """
        self.robot.relative_dynamics_factor = rel_vel
        success = False
        while not success:
            try:
                self.robot.move(motion, asynchronous=asynch)
                success = True
            except franky.ControlException as ex:
                rel_vel -= 0.02
                rel_vel = max(rel_vel, 0.02)
                self.robot.relative_dynamics_factor = rel_vel
                print(f"Encountered ControlException: {ex}")
                print("Attempting recovery with reduced dynamics...")
                time.sleep(1.0)
                self.robot.recover_from_errors()

    # ==========================#
    # Gripper functions         #
    # ==========================#
    def move_gripper(self, width: float, speed: float = None):
        success = False
        speed = speed or self._dflt_gripper_speed
        while not success:
            try:
                self.gripper.move(width, speed)
                success = True
            except franky.ControlException as ex:
                time.sleep(1.0)
                self.robot.recover_from_errors()
                speed -= 0.02
                speed = max(0.03, speed)

    def open_gripper(self, speed: float = None):
        success = False
        speed = speed or self._dflt_gripper_speed
        while not success:
            try:
                self.gripper.open(speed)
                success = True
            except franky.ControlException as ex:
                time.sleep(1.0)
                self.robot.recover_from_errors()
                speed -= 0.02
                speed = max(0.03, speed)

    def grasp(self, width: float = 0.0, force: float = None, speed: float = None):
        if force is not None:
            assert 0.0 <= force <= 50.0, "Gripper force must be between 0 and 50"
        success = False
        force = force or self._dflt_gripper_force
        speed = speed or self._dflt_gripper_speed
        while not success:
            try:
                self.gripper.grasp(width, speed, force, epsilon_outer=1.0)
                success = True
            except franky.CommandException as ex:
                time.sleep(1.0)
                self.robot.recover_from_errors()
                speed -= 0.02
                speed = max(0.03, speed)

    # ==========================#
    # Other functions          #
    # ==========================#
    def join_motion(self):
        self.robot.join_motion()

    def poll_motion(self):
        self.robot.poll_motion()

    @staticmethod
    def calculate_rotation_from_z_vector(z: np.ndarray) -> np.ndarray:
        z /= np.linalg.norm(z)  # normalize to length 1

        x_x = np.sqrt(1 / (1 + (z[0] ** 2 / z[2] ** 2)))
        x_z = -(z[0] / z[2]) * x_x
        x = np.array([x_x, 0, x_z])
        x /= np.linalg.norm(x)

        y = np.cross(z, x)
        y /= np.linalg.norm(y)

        rot = np.transpose(np.array([x, y, z]))
        rot = Rotation.from_matrix(rot)
        return rot.as_quat()


def test_panda(config: dict) -> None:
    robot = PandaReal(config)
    time.sleep(1)

    # open gripper and move to neutral
    robot.open_gripper()
    robot.move_to_neutral(rel_vel=0.05)

    # grasp and reopen to 4 cm
    robot.grasp()
    gripper_width = 0.04
    robot.move_gripper(width=gripper_width)
    if abs(robot.gripper_width - gripper_width) > 0.001:
        raise RuntimeError("PANDA TEST FAILED: gripper_width incorrect.")

    # move absolute
    abs_pos = np.array([0.5, -0.1, 0.3])
    robot.move_abs(abs_pos, rel_vel=0.05)
    if not np.isclose(abs_pos, robot.end_effector_position, atol=0.01).all():
        raise RuntimeError("PANDA TEST FAILED: absolute motion")

    # move relative
    current_pos = robot.end_effector_position
    rel_vec = np.array([-0.1, 0.1, 0.05])
    robot.move_rel(rel_vec, rel_vel=0.05)
    if not np.isclose(
        current_pos + rel_vec, robot.end_effector_position, atol=0.01
    ).all():
        print(abs_pos + rel_vec)
        raise RuntimeError("PANDA TEST FAILED: relative motion")

    # move relative in axis
    robot.move_in_axis_rel(Axis.X, -rel_vec[0], rel_vel=0.05)
    robot.move_in_axis_rel(Axis.Y, -rel_vec[1], rel_vel=0.05)
    robot.move_in_axis_rel(Axis.Z, -rel_vec[2], rel_vel=0.05)
    if not np.isclose(current_pos, robot.end_effector_position, atol=0.01).all():
        raise RuntimeError("PANDA TEST FAILED: relative motion in axis")

    # move absolute in axis
    x_pos = 0.4
    robot.move_in_axis_abs(Axis.X, value=x_pos, rel_vel=0.05)
    if not np.isclose(x_pos, robot.end_effector_position[0], atol=0.01).all():
        raise RuntimeError("PANDA TEST FAILED: absolute motion in X-axis")

    y_pos = -0.01
    robot.move_in_axis_abs(Axis.Y, value=y_pos, rel_vel=0.05)
    if not np.isclose(y_pos, robot.end_effector_position[1], atol=0.01).all():
        print(robot.end_effector_position)
        raise RuntimeError("PANDA TEST FAILED: absolute motion in Y-axis")

    z_pos = 0.4
    robot.move_in_axis_abs(Axis.Z, value=z_pos, rel_vel=0.05)
    if not np.isclose(z_pos, robot.end_effector_position[2], atol=0.01).all():
        raise RuntimeError("PANDA TEST FAILED: absolute motion in Z-axis")

    print("Panda test successful!")


def test_end_effector_state(config: dict) -> None:
    robot = PandaReal(config)
    rot = Rotation.from_quat(robot.end_effector_orientation)
    print(f"joints: {robot.current_joint_positions}")
    print(f"pos: {robot.end_effector_position}")
    print(f"ori: {rot.as_matrix()}")


def test_ende_effector_rotation(config: dict) -> None:
    robot = PandaReal(config)
    z_ref = np.array([0, 0, -1], dtype=np.float64)
    z = np.array(z_ref)

    # init
    rot_ref = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    print(rot_ref)
    rot_ref = Rotation.from_matrix(rot_ref)
    robot.move_rel(np.array([0, 0, 0]), 0.05, rot_ref.as_quat())

    for i in range(100):
        z_delta = np.zeros(3)
        z_delta[0] = 0.1 * random.random() - 0.05
        z_delta[1] = 0.1 * random.random() - 0.05

        z += z_delta  # np.array([-0.1, 0.1, 0])
        z /= np.linalg.norm(z)  # normalize to length 1
        alpha = np.arccos(np.dot(z, z_ref))

        rot = PandaReal.calculate_rotation_from_z_vector(z)
        robot.move_rel(np.array([0, 0, 0]), 0.05, rot)
        print(f"alpha: {np.degrees(alpha)}")

        if abs(np.degrees(alpha)) > 25:
            print("Rotation more than 25 degrees!")
            break


def test_neutral_pos(config: dict) -> None:
    robot = PandaReal(config)
    robot.move_to_neutral(0.05)


if __name__ == "__main__":
    with open("config_ip.yaml", "r") as f:
        config = yaml.safe_load(f)

    # test_panda(config)
    test_neutral_pos(config)
    test_end_effector_state(config)
    # test_ende_effector_rotation(config)