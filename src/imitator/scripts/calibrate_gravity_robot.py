from franky import *
import time
import numpy as np


if __name__ == "__main__":

    robot = Robot("10.10.10.10")

    robot.set_joint_impedance(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    #robot.relative_dynamics_factor = 0.05

    inertia = np.eye(3) * 1e-4
    inertia = inertia.T.reshape(-1)
    
    robot.set_load(1, [0.065, -0.056, 0.028], inertia.tolist())

    # center of mass roughly at [0.065, -0.056, 0.028]

    mjv1 = JointVelocityMotion(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        duration=Duration(1000000),
    )

    robot.move(mjv1)