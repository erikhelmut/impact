# impact

Imitation-driven Manipulation with Precise Adaptive Control and Tactile feedback


Learning Visuotactile Grasping Policies from Demonstration Data

ros2 launch foxglove_bridge foxglove_bridge_launch.xml address:=127.0.0.1


python lerobot/scripts/train.py \
    --policy.type=diffusion \
    --dataset.repo_id=/home/erik/impact_planting_task_v2_sub4 \
    --output_dir=/home/erik/impact/src/imitator/outputs/train/impact_planting_v2_sub4 \
    --wandb.enable=true