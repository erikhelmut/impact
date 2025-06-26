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

        # initialize the PandaReal instance
        with open("/home/erik/impact/src/franka_panda/config/panda_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        self.panda = PandaReal(config)

        # load calibration parameters for gripper
        self.m, self.c = np.load("/home/erik/impact/src/actuated_umi/calibration/20250526-133247.npy")
        #self.m, self.c = np.load("/home/erik/impact/src/actuated_umi/calibration/20250623-095223.npy")

        # create publisher to set goal force and gripper width of the actuated umi gripper
        self.imitator_publisher = self.create_publisher(GoalForceController, "set_actuated_umi_goal_force", 1)


        # configure LeRobotDataset for one or multiple episodes
        self.dataset = LeRobotDataset(
            repo_id=Path("/home/erik/impact_planting_task_new"),
            episodes=[1] # 1 seems kinda ok
        )

        # use standard PyTorch DataLoader to load the dataset
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            num_workers=8,
            batch_size=1,
            shuffle=False,
        )

        self.i = 200

        # get length of the dataset
        self.dataset_length = len(self.dataset)

        timer_period = 1.0 / 25  # 25 Hz
        self.timer = self.create_timer(timer_period, self.pub_state)


    def pub_state(self):

        try:

            batch = self.dataset[self.i]

            # extract the batch data
            state = batch["observation.state"]
            image = batch["observation.image"]
            action = batch["action"]

            goal_ee_pos = np.array(state.squeeze(0)[2:5].to("cpu").numpy())
            goal_ee_ori = np.array(state.squeeze(0)[5:9].to("cpu").numpy())

            # add 1cm of height to the goal position
            goal_ee_pos[2] += 0.01

            # move the robot arm
            self.panda.move_abs(
                goal_pos=goal_ee_pos,
                goal_ori=goal_ee_ori,
                rel_vel=0.07,
                asynch=True
            )

            msg = GoalForceController()
            #msg.goal_force = float(self.filt.filter(goal_force))
            msg.goal_force = float(0)
            msg.goal_position = int(self.m * state[1] + self.c -40)
            self.imitator_publisher.publish(msg)

            self.i += 1

            if self.i >= self.dataset_length -50:
                self.i = 100000
        
        except Exception as e:
            self.timer.cancel()
            # stop code
            exit(0)
    

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