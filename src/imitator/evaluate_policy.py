import sys; sys.path.append("/home/erik/impact/src/imitator/lerobot")

from pathlib import Path

import numpy
import torch
import matplotlib.pyplot as plt

from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy



# TODO: create following methods:
# - make predictions with the policy based on 2 observations
# - plot the predicted actions in comparison to the ground truth actions





if __name__ == "__main__":


    # Create a directory to store the video of the evaluation
    output_directory = Path("/home/erik/impact/src/imitator/outputs/eval/impact_diff_test")
    output_directory.mkdir(parents=True, exist_ok=True)

    # Select your device
    device = "cuda"

    # Provide the [hugging face repo id](https://huggingface.co/lerobot/diffusion_pusht):
    # pretrained_policy_path = "lerobot/diffusion_pusht"
    # OR a path to a local outputs/train folder.
    pretrained_policy_path = Path("/home/erik/impact/src/imitator/outputs/train/impact_diff_test/checkpoints/last/pretrained_model")

    policy = DiffusionPolicy.from_pretrained(pretrained_policy_path)


    # Verify the shapes of the features expected by the policy 
    print(policy.config.input_features)

    # Check the actions produced by the policy
    print(policy.config.output_features)

    # Reset the policy and environments to prepare for rollout
    policy.reset()

    # numpy_observation, info = env.reset(seed=42)

    # Prepare to collect every rewards and all the frames of the episode,
    # from initial state to final state.
    rewards = []
    frames = []

    # Render frame of the initial state
    #frames.append(env.render())


    # load one episode of my LeRobotDataset
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        repo_id=Path("/home/erik/erik_impact_dataset_from_hdf5"),
        episodes=[7]
    )


    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=1,
        shuffle=False,
    )


    gt_actions_0 = []
    p_actions_0 = []
    gt_actions_1 = []
    p_actions_1 = []
    i = 0


    for batch in dataloader:
        #print(f"{batch['observation.image'].shape=}")  # (32, 4, c, h, w)
        #print(f"{batch['observation.state'].shape=}")  # (32, 6, c)
        #print(f"{batch['action'].shape=}")  # (32, 64, c)

        
        state = batch["observation.state"]
        image = batch["observation.image"]
        action = batch["action"]


        gt_actions_0.append(action.squeeze(0)[0].to("cpu").numpy())
        gt_actions_1.append(action.squeeze(0)[1].to("cpu").numpy())

        # Send data tensors from CPU to GPU
        state = state.to(device, non_blocking=True)
        image = image.to(device, non_blocking=True)

        # Create the policy input dictionary
        observation = {
            "observation.state": state,
            "observation.image": image,
        }

        # Predict the next action with respect to the current observation
        with torch.inference_mode():
            policy_action = policy.select_action(observation)
            #print(f"{policy_action.shape=}")  # (32, 64, c)

        remaining_actions_in_queue = list(policy._queues["action"])
        print(len(remaining_actions_in_queue))

        # Prepare the action for the environment
        numpy_action_0 = policy_action.squeeze(0)[0].to("cpu").numpy()
        numpy_action_1 = policy_action.squeeze(0)[1].to("cpu").numpy()
        p_actions_0.append(numpy_action_0)
        p_actions_1.append(numpy_action_1)

        #print(f"{numpy_action.shape=}")  # (32, 64, c)

        i += 1

        #if i == 10:
        #    break




    # create a figure with 2 subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))

    axs[0].set_title("FEATS_FZ")
    axs[0].plot(gt_actions_0, label="gt_actions")
    axs[0].plot(p_actions_0, label="policy_actions")
    axs[0].set_ylim(-10, 1)
    axs[0].legend()
    axs[1].set_title("ARUCO_DIST")
    axs[1].plot(gt_actions_1, label="gt_actions")
    axs[1].plot(p_actions_1, label="policy_actions")
    axs[1].set_ylim(0, 600)
    axs[1].legend()


    plt.show()






    """
        Generation Phase (When Queue is Empty):

        When policy.select_action() is called AND its internal policy._queues["action"] is empty:

            The policy uses the current observation history (which has just been updated with the observation you passed in).

            It calls its internal diffusion model (self.diffusion.generate_actions(...)) to predict a sequence of future actions. This sequence is policy.config.horizon long.

            From this long sequence, it takes the first policy.config.n_action_steps actions.

            It populates policy._queues["action"] with these policy.config.n_action_steps actions. (Let's say n_action_steps is 8, so 8 actions go into the queue).

            It then popleft()s the very first action from this queue and returns it. The queue now contains 7 actions.

    Consumption Phase (When Queue is NOT Empty):

        When policy.select_action() is called AND policy._queues["action"] is NOT empty:

            The policy does NOT run the diffusion model again to predict new actions.

            It simply popleft()s the next available action from the policy._queues["action"] (which was filled during a previous Generation Phase) and returns that action.

            The observation you pass in during this phase is still used to update the observation history (policy._queues["observation.state"], etc.), but it's not immediately used to generate new actions unless the action queue becomes empty on this call.

    Loop Back to Generation:

        This consumption continues until policy._queues["action"] becomes empty.

        The very next time policy.select_action() is called (now with an empty action queue), it triggers the Generation Phase again (Point 1).

In short:

    Predicts n_action_steps (e.g., 8 actions).

    Returns the 1st one.

    For the next 7 calls to select_action, it just returns actions 2 through 8 from that initially predicted batch.

    Only after the 8th action is returned (and the queue is empty) will the next call to select_action trigger a new prediction of 8 actions.

This design is efficient for rollouts because:

    The action generation process (diffusion sampling) can be computationally more expensive than just taking an item from a queue.

    It allows the policy to have a short-term "plan" (the n_action_steps sequence) and execute it, re-planning only when that plan is exhausted.

So, your interpretation is spot on. If you want fresh predictions for every single observation, you need to intervene and clear that action queue before each call, as discussed.
    """