import numpy as np
import time

import traceback
from scipy.spatial.transform import Rotation
from transformation import Transformation
import yaml

from franka_panda.panda_real import PandaReal
from optitrack.optitrack import OptiTrack


def readStates(robot: PandaReal, ot: OptiTrack) -> list:
    """Read the neccessary states of the panda and calibration object to perform the calibration.

    Args:
        robot (PandaReal): object of the robot
        ot (OptiTrack): OptiTrack object

    Returns:
        list: of two tuples, first tuple -> end-effector state, second tuple -> state of the calibratio object, each tuple has two arrays, first array is the position, second array is the orientation
    """
    result = []
    result.append((robot.end_effector_position, robot.end_effector_orientation))
    result.append((ot.ee_pos, ot.ee_ori))
    return result


def transl(t):
    # TRANSL	Translational transform
    #
    # 	T= TRANSL(X, Y, Z)
    # 	T= TRANSL( [X Y Z] )
    #
    # 	[X Y Z]' = TRANSL(T)
    #
    # 	[X Y Z] = TRANSL(TG)
    #
    # 	Returns a homogeneous transformation representing a
    # 	translation of X, Y and Z.
    #
    # 	The third form returns the translational part of a
    # 	homogenous transform as a 3-element column vector.
    #
    # 	The fourth form returns a  matrix of the X, Y and Z elements
    # 	extracted from a Cartesian trajectory matrix TG.
    #
    r = np.eye(4)
    if len(t.shape) == 2:
        r[:-1, -1] = t[:, 0]
    else:
        r[:-1, -1] = t
    return r


def skew(V):
    # skew - returns skew matrix of a 3x1 vector.
    #        cross(V,U) = skew(V)*U
    #
    #    S = skew(V)
    #
    #          0  -Vz  Vy
    #    S =   Vz  0  -Vx
    #         -Vy  Vx  0
    #
    S = np.array(
        [[0, -V[2, 0], V[1, 0]], [V[2, 0], 0, -V[0, 0]], [-V[1, 0], V[0, 0], 0]]
    )
    return S


def rot2quat(R):
    # rot2quat - converts a rotation matrix (3x3) to a unit quaternion(3x1)
    #
    #    q = rot2quat(R)
    #
    #    R - 3x3 rotation matrix, or 4x4 homogeneous matrix
    #    q - 3x1 unit quaternion
    #        q = sin(theta/2) * v
    #        teta - rotation angle
    #        v    - unit rotation axis, |v| = 1
    #

    w4 = 2 * np.sqrt(1 + np.trace(R[0:3, 0:3]))  # can this be imaginary?
    q = np.array(
        [
            [(R[2, 1] - R[1, 2]) / w4],
            [(R[0, 2] - R[2, 0]) / w4],
            [(R[1, 0] - R[0, 1]) / w4],
        ]
    )
    return q


def quat2rot(q):
    # quat2rot - a unit quaternion(3x1) to converts a rotation matrix (3x3)
    #
    #    R = quat2rot(q)
    #
    #    q - 3x1 unit quaternion
    #    R - 4x4 homogeneous rotation matrix (translation component is zero)
    #        q = sin(theta/2) * v
    #        teta - rotation angle
    #        v    - unit rotation axis, |v| = 1
    #
    p = np.dot(q[:, 0], q[:, 0])
    if p > 1:
        print("Warning: quat2rot: quaternion greater than 1")

    w = np.sqrt(1 - p)  # w = cos(theta/2)

    R = np.eye(4)
    R[0:3, 0:3] = (
        2 * np.outer(q, q) + 2 * w * skew(q) + np.eye(3) - 2 * np.diag([p, p, p])
    )
    return R


def handEye(bHg, wHc):
    # handEye - performs hand/eye calibration
    #
    #     gHc = handEye(bHg, wHc)
    #
    #     bHg - pose of gripper relative to the robot base..
    #           (Gripper center is at: g0 = Hbg * [0;0;0;1] )
    #           Matrix dimensions are 4x4xM, where M is ..
    #           .. number of camera positions.
    #           Algorithm gives a non-singular solution when ..
    #           .. at least 3 positions are given
    #           Hbg(:,:,i) is i-th homogeneous transformation matrix
    #     wHc - pose of camera relative to the world ..
    #           (relative to the calibration block)
    #           Dimension: size(Hwc) = size(Hbg)
    #     gHc - 4x4 homogeneous transformation from gripper to camera
    #           , that is the camera position relative to the gripper.
    #           Focal point of the camera is positioned, ..
    #           .. relative to the gripper, at
    #                 f = gHc*[0;0;0;1];
    #
    # References: R.Tsai, R.K.Lenz "A new Technique for Fully Autonomous
    #           and Efficient 3D Robotics Hand/Eye calibration", IEEE
    #           trans. on robotics and Automaion, Vol.5, No.3, June 1989
    #
    # Notation: wHc - pose of camera frame (c) in the world (w) coordinate system
    #                 .. If a point coordinates in camera frame (cP) are known
    #                 ..     wP = wHc * cP
    #                 .. we get the point coordinates (wP) in world coord.sys.
    #                 .. Also refered to as transformation from camera to world
    #

    M = bHg.shape[2]

    K = (M * M - M) // 2  # Number of unique camera position pairs
    A = np.zeros((3 * K, 3))  # will store: skew(Pgij+Pcij)
    B = np.zeros((3 * K, 1))  # will store: Pcij - Pgij
    k = 0

    # Now convert from wHc notation to Hc notation used in Tsai paper.
    Hg = bHg
    # Hc = cHw = inv(wHc); We do it in a loop because wHc is given, not cHw
    Hc = np.zeros((4, 4, M))

    for i in range(M):
        Hc[:, :, i] = np.linalg.inv(wHc[:, :, i])

    for i in range(M):
        for j in range(i + 1, M):
            Hgij = np.dot(
                np.linalg.inv(Hg[:, :, j]), Hg[:, :, i]
            )  # Transformation from i-th to j-th gripper pose
            Pgij = 2 * rot2quat(Hgij)  # ... and the corresponding quaternion

            Hcij = np.dot(
                Hc[:, :, j], np.linalg.inv(Hc[:, :, i])
            )  # Transformation from i-th to j-th camera pose
            Pcij = 2 * rot2quat(Hcij)  # ... and the corresponding quaternion

            # Form linear system of equations
            A[k : (k + 3), 0:3] = skew(Pgij + Pcij)  # left-hand side
            B[k : (k + 3), :] = Pcij - Pgij  # right-hand side
            k += 3

    # Rotation from camera to gripper is obtained from the set of equations:
    #    skew(Pgij+Pcij) * Pcg_ = Pcij - Pgij
    # Gripper with camera is first moved to M different poses, then the gripper
    # .. and camera poses are obtained for all poses. The above equation uses
    # .. invariances present between each pair of i-th and j-th pose.

    Pcg_, residuals, rank, singulars = np.linalg.lstsq(A, B, rcond=None)
    Pcg_ = Pcg_[:, 0]

    # Pcg_ = A \ B;                # Solve the equation A*Pcg_ = B

    # Obtained non-unit quaternin is scaled back to unit value that
    # .. designates camera-gripper rotation
    Pcg = (2 * Pcg_ / np.sqrt(1 + np.dot(Pcg_, Pcg_)))[..., None]

    Rcg = quat2rot(Pcg / 2)  # Rotation matrix
    # Calculate translational component
    k = 0
    for i in range(M):
        for j in range(i + 1, M):
            Hgij = np.dot(
                np.linalg.inv(Hg[:, :, j]), Hg[:, :, i]
            )  # Transformation from i-th to j-th gripper pose
            Hcij = np.dot(
                Hc[:, :, j], np.linalg.inv(Hc[:, :, i])
            )  # Transformation from i-th to j-th camera pose

            # Form linear system of equations
            A[k : (k + 3), 0:3] = Hgij[0:3, 0:3] - np.eye(3)  # left-hand side
            B[k : (k + 3), 0] = (
                np.dot(Rcg[0:3, 0:3], Hcij[0:3, 3]) - Hgij[0:3, 3]
            )  # right-hand side
            k += 3

    Tcg, residuals, rank, singulars = np.linalg.lstsq(A, B, rcond=None)
    # Tcg = A \ B;
    gHc = np.dot(transl(Tcg), Rcg)  # incorporate translation with rotation

    return gHc


def pose_to_hom(trans, quat):
    """Pose (translation and quaternion) to homogeneous transformation matrix.

    Args:
        trans (np.ndarray): translation vector (3,)
        quat (np.ndarray): quaternion (4,)

    Returns:
        np.ndarray: homogeneous transformation matrix (4, 4)
    """
    if trans is not None and quat is not None:
        hom = np.zeros((4, 4))
        hom[3, 3] = 1
        rot_matrix = Rotation.from_quat(quat).as_matrix()
        hom[0:3, 0:3] = rot_matrix
        hom[0:3, 3] = trans
        return hom
    else:
        return None


def hom_to_pose(hom):
    """Homogeneous transformation matrix to pose (translation and quaternion).

    Args:
        hom (np.ndarray): homogeneous transformation matrix

    Returns:
        tuple: translation, quaternion
    """
    if hom is not None:
        trans = hom[0:3, 3]
        quat = Rotation.from_matrix(hom[0:3, 0:3]).as_quat()
        return trans, quat
    else:
        return None, None


class Calibrator:

    def __init__(self, min_num_samples):
        self.homMAs = np.empty((4, 4, 0))
        self.homMBs = np.empty((4, 4, 0))

        self.lastHomAM = None
        self.lastHomBM = None

        self.min_num_samples = min_num_samples

    def add_sample(self, homAM, homBM):
        self.homMAs = np.concatenate(
            (self.homMAs, np.linalg.inv(homAM)[..., np.newaxis]), axis=2
        )
        self.homMBs = np.concatenate(
            (self.homMBs, np.linalg.inv(homBM)[..., np.newaxis]), axis=2
        )

        self.lastHomAM = homAM
        self.lastHomBM = homBM

    def delete_sample(self, idx):
        if self.homMAs.shape[2] > 0:
            self.homMAs = np.delete(self.homMAs, idx, 2)
            self.homMBs = np.delete(self.homMBs, idx, 2)

        if self.homMAs.shape[2] > 0:
            self.lastHomAM = np.linalg.inv(self.homMAs[:, :, -1])
            self.lastHomBM = np.linalg.inv(self.homMBs[:, :, -1])
        else:
            self.lastHomAM = None
            self.lastHomBM = None

    def get_last_sample(self):
        return self.lastHomAM, self.lastHomBM

    def calibrate(self):
        print(f"Number of samples: {self.homMAs.shape[2]: d}")
        if self.homMAs.shape[2] >= self.min_num_samples:
            homAB = handEye(self.homMAs, self.homMBs)
            return homAB
        else:
            print(f"Waiting for at least {self.min_num_samples: d} samples")
            return None


class Collector:

    def __init__(self):
        self.transThreshold = 0.01
        self.rotThreshold = 0.9

    def reconfigure_callback(self, config, level):
        self.transThreshold = config["transThreshold"]
        self.rotThreshold = config["rotThreshold"]
        self.autoCollectInterval = config["autoCollectRate"]

    def checkPoseDifference(self, posA, quatA, posB, quatB):
        diffPos = np.sqrt(np.sum((np.array(posA) - np.array(posB)) ** 2))
        diffQuat = np.absolute(np.dot(np.array(quatA), np.array(quatB)))
        return diffPos >= self.transThreshold or diffQuat <= self.rotThreshold

    def collect_sample(
        self,
        sample: list,
        lastTransAM=None,
        lastQuatAM=None,
        lastTransBM=None,
        lastQuatBM=None,
    ):
        (transAM, quatAM) = sample[0]
        if lastTransAM is not None and lastQuatAM is not None:
            assert self.checkPoseDifference(
                transAM, quatAM, lastTransAM, lastQuatAM
            ), "Difference from the last sample is not big enough."

        (transBM, quatBM) = sample[1]
        if lastTransBM is not None and lastQuatBM is not None:
            assert self.checkPoseDifference(
                transBM, quatBM, lastTransBM, lastQuatBM
            ), "Difference from the last sample is not big enough."

        return transAM, quatAM, transBM, quatBM


class Core:

    def __init__(self):
        self.collector = Collector()
        self.calibrator = Calibrator(min_num_samples=3)

        self.transAB = [0.0, 0.0, 0.0]
        self.quatAB = [0.0, 0.0, 0.0, 1.0]

    def reconfigure_callback(self, config, level):
        print("reconfigure cb\n", config)

        self.collector.reconfigure_callback(config, level)
        self.calibrator.reconfigure_callback(config, level)

        return config

    def add_sample(self, sample: list):
        print("Adding a transform sample\n")
        try:
            lastHomAM, lastHomBM = self.calibrator.get_last_sample()

            lastTransAM, lastQuatAM = hom_to_pose(lastHomAM)
            lastTransBM, lastQuatBM = hom_to_pose(lastHomBM)

            transAM, quatAM, transBM, quatBM = self.collector.collect_sample(
                sample, lastTransAM, lastQuatAM, lastTransBM, lastQuatBM
            )

            homAM = pose_to_hom(transAM, quatAM)
            homBM = pose_to_hom(transBM, quatBM)

            self.calibrator.add_sample(homAM, homBM)
            homAB = self.calibrator.calibrate()

            if homAB is not None:
                self.transAB, self.quatAB = hom_to_pose(homAB)
                print(
                    f"New transformation (xyz/xzyw): "
                    f"{self.transAB[0] :.5f} {self.transAB[1] :.5f} {self.transAB[2] :.5f} / "
                    f"{self.quatAB[0] :.5f} {self.quatAB[1] :.5f} {self.quatAB[2] :.5f} {self.quatAB[3] :.5f}"
                )
            else:
                print("No new transformation")
        except:
            raise

    def delete_sample(self, idx):
        print(f"deleting sample {idx :d}")

        self.calibrator.delete_sample(idx)
        homAB = self.calibrator.calibrate()
        if homAB is not None:
            self.transAB = homAB[0:3, 3]
            self.quatAB = Rotation.from_matrix(homAB[0:3, 0:3]).as_quat()
            print(
                f"New transformation (xyz/xzyw): "
                f"{self.transAB[0] :.5f} {self.transAB[1] :.5f} {self.transAB[2] :.5f} / "
                f"{self.quatAB[0] :.5f} {self.quatAB[1] :.5f} {self.quatAB[2] :.5f} {self.quatAB[3] :.5f}"
            )
        else:
            print("No new transformation")

    def get_transformation(self):
        return self.transAB, self.quatAB


def calibrate_manual(panda: PandaReal, ot: OptiTrack) -> Transformation:
    """Manula calibration where the user has to move the robot's end-effector manually to new positions.

    Args:
        panda (PandaReal): object of the robot
        ot (OptiTrack): OptiTrack object

    Returns:
        Transformation: transformation from OptiTrack to panda coordinates
    """
    core = Core()
    saved_points = 0
    for i in range(100):
        print(f"Waypoint {i} - Move the end-effector to a new position!")
        response = input("Do you want to use the current sample or not? y/n")
        result = readStates(panda, ot)
        if response == "y" or response == "Y":
            print("using current sample")
            print(f"Joint positions: {panda.current_joint_positions}")
            core.add_sample(result)
            saved_points += 1
        else:
            print("skipping current sample")
        if saved_points >= 3:
            print(f"{saved_points} waypoints are saved!")
            response = input("Do you want to continue sampling waypoints? y/n")
            if response == "y" or response == "Y":
                continue
            else:
                break

    # Estimate the transforms using the recorded positions
    trans, quat = core.get_transformation()
    return Transformation.from_pos_quat(trans, quat)


def calibrate(
    panda: PandaReal, ot: OptiTrack, waypoints: list[list], rel_vel=0.05
) -> Transformation:
    """Calibration where the robot moves automatically to new positions.

    Args:
        panda (PandaReal): object of the robot
        ot (OptiTrack): OptiTrack object
        waypoints (list[list]): list of joint positions the robot should move to (at least 3 neccessary!)
        rel_vel (float, optional): relative velocity of the robot during the calibration process. Defaults to 0.05.

    Returns:
        Transformation: transformation from OptiTrack to panda coordinates
    """
    assert len(waypoints) >= 3, "At least 3 waypoints are neccessary!"
    print(f"Calibration with {len(waypoints)} waypoints.")
    core = Core()
    for w in waypoints:
        panda.move_to_joint_position(w, rel_vel)
        time.sleep(0.5)
        result = readStates(panda, ot)
        core.add_sample(result)

    # Estimate the transforms using the recorded positions
    trans, quat = core.get_transformation()
    return Transformation.from_pos_quat(trans, quat)


def main():
    # Load config and initialize panda and OptiTrack
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    panda = PandaReal(config)
    ot = OptiTrack()
    calibrations = 6
    transformation = np.zeros((calibrations, 4, 4))

    # Do calibration
    # transformation = calibrate_manual(panda, ot)
    for cal in range(calibrations):
        transformation[cal, :, :] = calibrate(
            panda,
            ot,
            config["waypoints"],
            config["calibration_velocity"],
        ).matrix
    mean = np.mean(transformation, axis=0)
    var = np.var(transformation, axis=0)
    print(f"Mean: {mean}")
    print(f"Variance: {var}")
    ot.transformation = mean
    np.save("calibration.npy", mean)

    ##############################################################
    # Save the transformation from optitrack to panda end-effector

    from scipy.spatial.transform import Rotation as R
    
    ee_pos_ot = ot.ee_pos
    ee_ori_ot = ot.ee_ori

    ee_pos_panda = panda.end_effector_position
    ee_ori_panda = panda.end_effector_orientation

    # convert optitrack transformation to panda coordinates
    # extract rotation from transformation matrix
    R_T = transformation[:3, :3]

    # convert to quaternion
    q_T = R.from_matrix(R_T).as_quat()

    # Rotation transformieren
    rot_T = R.from_quat(q_T)
    rot_ee = R.from_quat(ee_ori_ot)
    ee_ori_ot_panda = (rot_T * rot_ee).as_quat()

    # transform end-effector position
    ee_pos_ot_panda = transformation[:3, :3] @ ee_pos_ot + transformation[:3, 3]

    # calculate the transformation from optitrack to panda end-effector
    
    translate_ot_panda = ee_pos_panda - ee_pos_ot_panda

    # needed: relative rotation from optitrack to panda end-effector
    rot_ot_panda = R.from_quat(ee_ori_ot_panda).inv() * R.from_quat(ee_ori_panda)
    rot_ori_ot_panda = rot_ot_panda.as_quat()

    # OT^R_EE = FB^R_OT^(-1) * FB^R_EE


    ##############################################################

    ot.close()


if __name__ == "__main__":
    main()
