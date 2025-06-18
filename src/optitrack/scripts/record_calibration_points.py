from franky import *
import numpy as np
import time


if __name__ == "__main__":
    """
    Simple script to record calibration points for the Franka Panda robot using OptiTrack.
    """

    robot = Robot("10.10.10.10")

    # set joint impedance to zero
    robot.set_joint_impedance(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    inertia = np.eye(3) * 1e-4
    inertia = inertia.T.reshape(-1)
    
    # center of mass roughly at [0.065, -0.056, 0.028]
    robot.set_load(1, [0.065, -0.056, 0.028], inertia.tolist())

    p = 12
    joints_states = []
    for i in range(p):
        print(f"Recording calibration point {i + 1}/{p}. Please move the robot to the desired position.")
        retval = input("Press Enter to record the current joint states...")
        joints_state = robot.current_joint_state.position.tolist()
        joints_state = robot.current_joint_state.position.tolist()
        joints_states.append(joints_state)
        print(f"Recorded joint states: {joints_state}\n")

    # remove the first sample as it is usually the initial position
    joints_states = joints_states[1:]

    # print recorded joint states
    print(joints_states)
