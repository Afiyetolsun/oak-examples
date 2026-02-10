#!/usr/bin/env bash
set -e

echo "Listing workspace:"
ls -la /ws

echo "Sourcing ROS workspace:"
source /ws/install/setup.bash

echo "Starting depthai launch..."
ros2 launch depthai_filters spatial_bb.launch.py &
LAUNCH_PID=$!

echo "Starting follow_person node..."
ros2 run follow_person follow_person_node

# If follow_person exits, stop launch as well
echo "follow_person exited, stopping launch..."
kill -TERM "$LAUNCH_PID"
wait "$LAUNCH_PID"