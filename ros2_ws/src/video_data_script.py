'''
source /opt/ros/humble/setup.bash
source /workspace/ros2_ws/install/setup.bash
python3 coverage_node.py
'''

#!/usr/bin/env python3

import math
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


def ask_limo_number():
    return "/limo" + input(
        "Enter the last 3 digits of the LIMO: "
    ).strip() + "/cmd_vel"


class CoverageDriver(Node):

    def __init__(self, topic):

        super().__init__("coverage_driver")

        self.publisher = self.create_publisher(
            Twist,
            topic,
            10,
        )

        # -----------------------------
        # Robot parameters
        # -----------------------------
        self.forward_speed = 0.20      # m/s
        self.rotation_speed = 0.50     # rad/s

        # -----------------------------
        # Workspace
        # -----------------------------
        self.width = 1.0               # meters
        self.height = 1.5              # meters

        self.cell_spacing = 0.15       # meters

        # -----------------------------
        # Timing
        # -----------------------------
        self.pause_time = 1.0

        self.cell_time = (
            self.cell_spacing /
            self.forward_speed
        )

        self.turn90_time = (
            (math.pi / 2) /
            self.rotation_speed
        )

        self.turn360_time = (
            (2 * math.pi) /
            self.rotation_speed
        )

        # Number of cells

        self.cols = int(round(
            self.width /
            self.cell_spacing
        ))

        self.rows = int(round(
            self.height /
            self.cell_spacing
        ))

        self.direction = 1      # 1=east, -1=west

        self.get_logger().info(
            f"Coverage: {self.rows+1} rows x {self.cols+1} cols"
        )

    def publish_cmd(self, linear, angular):

        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular
        self.publisher.publish(msg)

    def run_for(self, linear, angular, duration):

        start = time.time()

        while time.time() - start < duration:

            self.publish_cmd(linear, angular)
            rclpy.spin_once(self, timeout_sec=0.02)

        self.stop()

    def stop(self):

        for _ in range(5):
            self.publish_cmd(0.0, 0.0)
            rclpy.spin_once(self, timeout_sec=0.02)

    def pause(self):

        self.run_for(
            0.0,
            0.0,
            self.pause_time,
        )

    def drive_one_cell(self):

        self.get_logger().info("Drive")

        self.run_for(
            self.forward_speed,
            0.0,
            self.cell_time,
        )

    def rotate360(self):

        self.get_logger().info("360 scan")

        self.run_for(
            0.0,
            self.rotation_speed,
            self.turn360_time,
        )

    def turn_left(self):

        self.run_for(
            0.0,
            self.rotation_speed,
            self.turn90_time,
        )

    def turn_right(self):

        self.run_for(
            0.0,
            -self.rotation_speed,
            self.turn90_time,
        )

    def next_row(self):

        if self.direction == 1:

            self.turn_left()

            self.drive_one_cell()

            self.turn_left()

        else:

            self.turn_right()

            self.drive_one_cell()

            self.turn_right()

        self.direction *= -1

    def run(self):

        try:

            for row in range(self.rows + 1):

                for col in range(self.cols + 1):

                    if not rclpy.ok():
                        return

                    # Skip the first move
                    if not (row == 0 and col == 0):
                        self.drive_one_cell()

                    self.pause()

                    self.rotate360()

                    self.pause()

                if row != self.rows:

                    self.next_row()

                    self.pause()

                    self.rotate360()

                    self.pause()

            self.get_logger().info("Coverage complete!")

        except KeyboardInterrupt:
            pass

        finally:

            self.stop()

            for _ in range(20):
                self.stop()

            self.get_logger().info("Stopped.")

def main():

    rclpy.init()

    topic = ask_limo_number()

    node = CoverageDriver(topic)

    try:
        node.run()

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received.")

    finally:
        # Send multiple stop commands before exiting
        for _ in range(20):
            node.stop()

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()