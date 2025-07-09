#!/bin/bash

# Define variables
RECORD_DIR="/media/erik/IAS_ERIK/bag_files/planting_task_real_dirt/$(date +%Y-%m-%d_%H-%M-%S)"
LAUNCH_FILE="impact umi_feats_launch.py"

# Start the ROS 2 launch file
echo "Launching: $LAUNCH_FILE"
ros2 launch $LAUNCH_FILE &

# Save the PID of the launch process
LAUNCH_PID=$!

# Start ros2 bag recording in the background
echo "Starting ros2 bag recording in: $RECORD_DIR"
ros2 bag record --all -o "$RECORD_DIR" &

# Save the PID of the background process
BAG_PID=$!

# Wait for the launch process to finish
wait $LAUNCH_PID

# When the launch process exits, kill the ros2 bag process
echo "Shutting down ros2 bag recording..."
kill $BAG_PID

echo "Done."
