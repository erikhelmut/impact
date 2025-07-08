#!/usr/bin/env python3
import sys
from pathlib import Path

import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import plotly.graph_objects as go

def get_episode_trajectory(dataset_path, episode):
    """
    Load one episode and return lists of EE_POS_X, EE_POS_Y, EE_POS_Z from the ground-truth actions.
    """
    dataset = LeRobotDataset(
        repo_id=Path(dataset_path),
        episodes=[episode]
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    xs, ys, zs = [], [], []
    for batch in loader:
        action = batch["action"].squeeze(0)  # shape [action_dim]
        # indices 2,3,4 correspond to EE_POS_X, EE_POS_Y, EE_POS_Z
        xs.append(action[2].item())
        ys.append(action[3].item())
        zs.append(action[4].item())
    return xs, ys, zs

def main(dataset_path, num_episodes=50):
    # Collect all trajectories
    all_trajs = []
    subsample = 4
    for ep in range(num_episodes):
        xs, ys, zs = get_episode_trajectory(dataset_path, ep)
        xs = xs[::subsample]
        ys = ys[::subsample]
        zs = zs[::subsample]
        all_trajs.append((xs, ys, zs))
    
    # Build Plotly figure
    fig = go.Figure()
    for ep, (xs, ys, zs) in enumerate(all_trajs):
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode='lines',
                name=f'Episode {ep}'
            )
        )

        idxs = range(0, len(xs), 16)
        fig.add_trace(go.Scatter3d(
            x=[xs[i] for i in idxs],
            y=[ys[i] for i in idxs],
            z=[zs[i] for i in idxs],
            mode='markers',
            marker=dict(size=4),
            showlegend=False
        ))

    fig.update_layout(
        title='3D End-Effector Trajectories',
        scene=dict(
            xaxis_title='EE_POS_X',
            yaxis_title='EE_POS_Y',
            zaxis_title='EE_POS_Z'
        ),
        legend=dict(itemsizing='trace')
    )

    # This will open in your default browser with full mouse/gesture controls
    fig.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot 3D EE trajectories for all episodes (Plotly)")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/home/erik/impact_planting_task_v2",
        help="Path to the LeRobot dataset root"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=50,
        help="Number of episodes to plot (0 through num_episodes-1)"
    )
    args = parser.parse_args()
    main(args.dataset_path, args.num_episodes)
