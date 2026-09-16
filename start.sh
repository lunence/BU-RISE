#!/bin/bash

# Start camera streamer in the background
python3 camera_stream/stream_camera.py &
python3 camera_stream/display_server.py &
CAMERA_PID=$!

# Stop camera streamer when this script exits
trap "kill $CAMERA_PID" EXIT


# Start the dev container
docker compose up -d

# Run your ROS launch file inside the container
docker exec -it sad_volhard bash -c "
colcon build
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch localization localization_bringup.launch.py
"

# command:
# chmod +x start.sh
# ./start.sh