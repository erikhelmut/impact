from franky import *
import time
import numpy as np


if __name__ == "__main__":
    """
    Simple script to calibrate the gravity compansation of the Franka Panda robot.
    """

    robot = Robot("10.10.10.10")

    # set joint impedance to zero
    robot.set_joint_impedance(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    inertia = np.eye(3) * 1e-4
    inertia = inertia.T.reshape(-1)
    
    # center of mass roughly at [0.065, -0.056, 0.028]
    robot.set_load(1, [0.065, -0.056, 0.028], inertia.tolist())

    # start joint velocity mode with zero velocity
    mjv1 = JointVelocityMotion(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        duration=Duration(1000000),
    )

    robot.move(mjv1)