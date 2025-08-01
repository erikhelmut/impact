# impact

Imitation-driven Manipulation with Precise Adaptive Control and Tactile feedback


Learning Visuotactile Grasping Policies from Demonstration Data

ros2 launch foxglove_bridge foxglove_bridge_launch.xml address:=127.0.0.1


python3 lerobot/scripts/train.py \
    --policy.type=diffusion \
    --dataset.repo_id=/home/erik/impact_grape_task \
    --output_dir=/home/erik/impact/src/imitator/outputs/train/impact_grape \
    --wandb.enable=true \
    --steps=60000