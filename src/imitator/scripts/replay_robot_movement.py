import sys; sys.path.append("/home/erik/impact/src/imitator/lerobot")

from pathlib import Path
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt

import yaml
from franka_panda.panda_real import PandaReal

from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def plot_actions(gt_actions, p_actions):
    """
    Plot the ground truth and predicted actions for the two features.

    :param gt_actions: list of ground truth actions
    :param p_actions: list of predicted actions
    :return: None
    """

    # create figure
    fig, axs = plt.subplots(9, 1, figsize=(10, 18))
    plt.subplots_adjust(hspace=0.5)

    # tight layout
    plt.tight_layout()

    # plot gt_actions and p_actions
    axs[0].set_title("FEATS_FZ")
    axs[0].plot(gt_actions[0], label="gt_actions")
    axs[0].plot(p_actions[0], label="policy_actions")
    axs[0].set_ylim(-10, 1)
    axs[0].legend()

    axs[1].set_title("ARUCO_DIST")
    axs[1].plot(gt_actions[1], label="gt_actions")
    axs[1].plot(p_actions[1], label="policy_actions")
    axs[1].set_ylim(0, 600)
    axs[1].legend()

    axs[2].set_title("EE_POS_X")
    axs[2].plot(gt_actions[2], label="gt_actions")
    axs[2].plot(p_actions[2], label="policy_actions")
    #axs[2].set_ylim(-0.5, 0.5)
    axs[2].legend()

    axs[3].set_title("EE_POS_Y")
    axs[3].plot(gt_actions[3], label="gt_actions")
    axs[3].plot(p_actions[3], label="policy_actions")
    #axs[3].set_ylim(-0.5, 0.5)
    axs[3].legend()

    axs[4].set_title("EE_POS_Z")
    axs[4].plot(gt_actions[4], label="gt_actions")
    axs[4].plot(p_actions[4], label="policy_actions")
    #axs[4].set_ylim(-0.5, 0.5)
    axs[4].legend()

    axs[5].set_title("EE_ORI_X")
    axs[5].plot(gt_actions[5], label="gt_actions")
    axs[5].plot(p_actions[5], label="policy_actions")
    #axs[5].set_ylim(-0.5, 0.5)
    axs[5].legend()

    axs[6].set_title("EE_ORI_Y")
    axs[6].plot(gt_actions[6], label="gt_actions")
    axs[6].plot(p_actions[6], label="policy_actions")
    #axs[6].set_ylim(-0.5, 0.5)
    axs[6].legend()

    axs[7].set_title("EE_ORI_Z")
    axs[7].plot(gt_actions[7], label="gt_actions")
    axs[7].plot(p_actions[7], label="policy_actions")
    #axs[7].set_ylim(-0.5, 0.5)
    axs[7].legend()

    axs[8].set_title("EE_ORI_W")
    axs[8].plot(gt_actions[8], label="gt_actions")
    axs[8].plot(p_actions[8], label="policy_actions")
    #axs[8].set_ylim(-0.5, 0.5)
    axs[8].legend()

    plt.show()


def compute_rollout(policy, dataloader, device):
    """
    Compute the rollout of the policy for one or multiple episodes.

    :param policy: the policy to evaluate
    :param dataloader: the dataloader for the dataset
    :param device: the device to use for computation
    :return gt_actions: list of ground truth actions
    :return p_actions: list of predicted actions
    """

    # store the ground truth and predicted actions
    gt_actions = [[], [], [], [], [], [], [], [], []]
    p_actions = [[], [], [], [], [], [], [], [], []]

    for batch in dataloader:
        
        # extract the batch data
        state = batch["observation.state"]
        image = batch["observation.image"]
        action = batch["action"]

        # store the ground truth actions
        gt_actions[0].append(action.squeeze(0)[0].to("cpu").numpy())
        gt_actions[1].append(action.squeeze(0)[1].to("cpu").numpy())
        gt_actions[2].append(action.squeeze(0)[2].to("cpu").numpy())
        gt_actions[3].append(action.squeeze(0)[3].to("cpu").numpy())
        gt_actions[4].append(action.squeeze(0)[4].to("cpu").numpy())
        gt_actions[5].append(action.squeeze(0)[5].to("cpu").numpy())
        gt_actions[6].append(action.squeeze(0)[6].to("cpu").numpy())
        gt_actions[7].append(action.squeeze(0)[7].to("cpu").numpy())
        gt_actions[8].append(action.squeeze(0)[8].to("cpu").numpy())

        # send data tensors from CPU to GPU
        state = state.to(device, non_blocking=True)
        image = image.to(device, non_blocking=True)

        # create the policy input dictionary
        observation = {
            "observation.state": state,
            "observation.image": image,
        }

        # predict the next action with respect to the current observation
        # the policy internaly handles the queue of observations and actions
        with torch.inference_mode():
            policy_action = policy.select_action(observation)

        # get the remaining actions in the queue
        # remaining_actions_in_queue = list(policy._queues["action"])

        # store the predicted actions
        p_actions[0].append(policy_action.squeeze(0)[0].to("cpu").numpy())
        p_actions[1].append(policy_action.squeeze(0)[1].to("cpu").numpy())
        p_actions[2].append(policy_action.squeeze(0)[2].to("cpu").numpy())
        p_actions[3].append(policy_action.squeeze(0)[3].to("cpu").numpy())
        p_actions[4].append(policy_action.squeeze(0)[4].to("cpu").numpy())
        p_actions[5].append(policy_action.squeeze(0)[5].to("cpu").numpy())
        p_actions[6].append(policy_action.squeeze(0)[6].to("cpu").numpy())
        p_actions[7].append(policy_action.squeeze(0)[7].to("cpu").numpy())
        p_actions[8].append(policy_action.squeeze(0)[8].to("cpu").numpy())

    return gt_actions, p_actions


def main(pretrained_policy_path, dataset_path, episodes):
    """
    Main function to evaluate the policy.

    :param pretrained_policy_path: path to the pretrained policy
    :param dataset_path: path to the dataset
    :param episodes: list of episodes to evaluate
    :return: None
    """

    # initialize the PandaReal instance
    with open("/home/erik/impact/src/franka_panda/config/panda_config.yaml", "r") as f:
            config = yaml.safe_load(f)
    panda = PandaReal(config)
    
    # select your device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # provide the hugging face repo id or path to a local outputs/train folder
    #pretrained_policy_path = Path(pretrained_policy_path)

    # initialize the policy
    #policy = DiffusionPolicy.from_pretrained(pretrained_policy_path)

    # verify the shapes of the features expected by the policy 
    #print(policy.config.input_features)
    # check the actions produced by the policy
    #print(policy.config.output_features)

    # reset the policy to prepare for rollout
    #policy.reset()

    # configure LeRobotDataset for one or multiple episodes
    dataset = LeRobotDataset(
        repo_id=Path(dataset_path),
        episodes=episodes
    )

    # use standard PyTorch DataLoader to load the dataset
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=1,
        shuffle=False,
    )

    i = 0
    for batch in dataloader:

        i += 1
        
        if i > 200:

            # extract the batch data
            state = batch["observation.state"]
            image = batch["observation.image"]
            action = batch["action"]

            goal_ee_pos = np.array(state.squeeze(0)[2:5].to("cpu").numpy())
            goal_ee_ori = np.array(state.squeeze(0)[5:9].to("cpu").numpy())

            # move the robot arm
            panda.move_abs(
                goal_pos=goal_ee_pos,
                goal_ori=goal_ee_ori,
                rel_vel=0.01,
                asynch=False
            )

    # compute the rollout of the policy
    #gt_actions, p_actions = compute_rollout(policy, dataloader, device)
    
    # plot the actions
    #plot_actions(gt_actions, p_actions)




if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description="Evaluate the policy")
    parser.add_argument(
        "--pretrained_policy_path",
        type=str,
        default="/home/erik/impact/src/imitator/outputs/train/impact_planting/checkpoints/last/pretrained_model",
        help="Path to the pretrained policy",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/home/erik/impact_planting_task",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=[7],
        help="List of episodes to evaluate",
    )
    args = parser.parse_args()

    main(args.pretrained_policy_path, args.dataset_path, args.episodes)