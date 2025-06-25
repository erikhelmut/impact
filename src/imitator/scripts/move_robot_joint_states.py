import sys; sys.path.append("/home/erik/impact/src/imitator/lerobot")

from pathlib import Path
import argparse

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from impact_msgs.msg import GoalForceController

import numpy as np
import torch
import matplotlib.pyplot as plt

import time
import yaml
from franka_panda.panda_real import PandaReal

from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


class ReplayNode(Node):

    def __init__(self):
        """
        This node is responsible for replaying the robot movements
        based on the recorded dataset.

        :return: None
        """
        
        super().__init__("replay_node")

        self.waypoints = [[0.18559173983038824, 0.16621096462962828, 0.08209514929343502, -2.418032919551159, -0.2206065144965331, 2.3598716132402524, 0.575407437874225], [0.09569335967341286, -0.02051256604929194, 0.01801391964764751, -2.333044864251401, -0.2206247189220781, 2.1289841931375255, 0.5990102715688306], [-0.0383048028764414, -0.01580260732333728, -0.008777657579371404, -2.216367283317764, -0.21953061851559968, 2.0068554288626665, 0.6090536489457773], [-0.08330114575137677, 0.31128547280771496, -0.01022410667099209, -1.7688639197521496, -0.21751258963483555, 1.8401974765099354, 0.6312878700362906], [-0.08074980563689957, 0.5534834040515649, -0.005964880482498639, -1.5741439262434118, -0.20053209248953685, 1.8816444840007027, 0.6360097327109127], [-0.0859569800491615, 0.6358105069707286, -0.0060432219404392045, -1.5843997846817084, -0.17517509916284194, 1.9793274678088044, 0.6638761625327583], [-0.07929580896170293, 0.28902247278145454, 0.00838383686356049, -2.222750008609099, -0.15728362027050566, 2.4933736746049453, 0.842380422441232], [-0.0014802029168152122, -0.11144859864682007, 0.011899863510580963, -3.019737841503461, -0.16080315728820874, 2.8667296120793297, 0.6083556874203757]]

        # initialize the PandaReal instance
        with open("/home/erik/impact/src/franka_panda/config/panda_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        self.panda = PandaReal(config)

        # load calibration parameters for gripper
        self.m, self.c = np.load("/home/erik/impact/src/actuated_umi/calibration/20250526-133247.npy")
        #self.m, self.c = np.load("/home/erik/impact/src/actuated_umi/calibration/20250623-095223.npy")

        self.i = 0

        timer_period = 1.0 / 1  # 25 Hz
        self.timer = self.create_timer(timer_period, self.pub_state)


    def pub_state(self):

        try:

            joint_state = self.waypoints[self.i]

            self.panda.move_to_joint_position(
                joint_positions=joint_state,
                rel_vel=0.05,
                asynch=False
            )

            #msg = GoalForceController()
            #msg.goal_force = float(self.filt.filter(goal_force))
            #msg.goal_force = float(0)
            #msg.goal_position = int(self.m * state[1] + self.c + 50)
            #self.imitator_publisher.publish(msg)

            self.i += 1
        
        except Exception as e:
            print("Demo done.")
            # stop the timer
            self.timer.cancel()
    

def main(args=None):

    try:

        rclpy.init(args=args)

        replay_node = ReplayNode()

        rclpy.spin(replay_node)

    finally:

        replay_node.destroy_node()
        rclpy.shutdown()



if __name__ == "__main__":

    main()
