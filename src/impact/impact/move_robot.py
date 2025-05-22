from franky import *
import time
import numpy as np


class Roboter:
    
    def __init__(self, ip="10.10.10.10"):
        self.robot = self._connect(ip)
        self._configure_robot()

    def _connect(self, ip):
        return Robot(ip)

    def _configure_robot(self):
        self.robot.relative_dynamics_factor = 0.1
        inertia = np.eye(3) * 1e-4
        inertia = inertia.T.reshape(-1)
        self.robot.set_load(2.0, [0.0, 0.0, 0.1], inertia.tolist())


    def follow_trajectory(self):
        
        m_jp1 = JointWaypointMotion([
            JointWaypoint([0.122237, 0.183304, -0.0096206, -2.98456, 0.173523, 3.17774, 0.712785]),
            JointWaypoint([0.143187, 0.383379, -0.0877353, -2.46124, 0.0415854, 2.67903, 0.79327]),
        ])

        m_jp1e = JointStopMotion()

        m_cp1 = CartesianWaypointMotion([
            CartesianWaypoint(Affine([0.0, 0.0, -0.02]), ReferenceType.Relative)
        ])

        m_cp1e = CartesianStopMotion()

        m_jp2 = JointWaypointMotion([
            JointWaypoint([0.120061, 0.401312, -0.0806885, -2.09574, -0.0104175, 2.29513, 0.826725]),
        ])

        m_jp22 = JointWaypointMotion([
            JointWaypoint([0.111976, 0.718294, -0.087823, -1.6334, 0.0533854, 2.16891, 0.798216]),
        ])

        m_jp3 = JointWaypointMotion([
            JointWaypoint([0.122237, 0.183304, -0.0096206, -2.98456, 0.173523, 3.17774, 0.712785]),
        ])

        self.robot.move(m_jp1)
        self.robot.move(m_jp1e)
        input("Press Enter to continue...")
        
        self.robot.move(m_cp1, asynchronous=True)
        time.sleep(0.2)
        self.robot.move(CartesianStopMotion())
        self.robot.move(m_jp2, asynchronous=True)
        time.sleep(1.2)
        self.robot.move(m_jp22)
        input("Press Enter to continue...")
        self.robot.move(m_jp3)
        self.robot.move(m_cp1e)



if __name__ == "__main__":
    roboter = Roboter()
    roboter.follow_trajectory()