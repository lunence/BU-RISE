from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    return LaunchDescription([

        # Localization node
        Node(
            package="localization",
            executable="localization_node",
            name="localization",
            output="screen"
        ),

        # Pure pursuit node
        Node(
            package="pure_pursuit",
            executable="pure_pursuit_node",
            name="pure_pursuit",
            output="screen"
        ),

    ])

# cd /workspace/ros2_ws
# colcon build --symlink-install
# source install/setup.bash
# ros2 launch localization localization_bringup.launch.py