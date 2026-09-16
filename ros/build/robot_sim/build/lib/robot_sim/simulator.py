import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, PoseStamped

import math


class RobotSimulator(Node):

    def __init__(self):
        super().__init__("robot_simulator")

        # robot state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        self.v = 0.0
        self.w = 0.0

        self.dt = 0.05

        self.cmd_sub = self.create_subscription(
            Twist,
            "/limo000/cmd_vel",
            self.cmd_callback,
            10
        )

        self.pose_pub = self.create_publisher(
            PoseStamped,
            "/pred_pose",
            10
        )

        self.timer = self.create_timer(
            self.dt,
            self.update
        )


    def cmd_callback(self, msg):

        self.v = msg.linear.x
        self.w = msg.angular.z


    def update(self):

        # differential drive model
        self.x += self.v * math.cos(self.theta) * self.dt
        self.y += self.v * math.sin(self.theta) * self.dt
        self.theta += self.w * self.dt


        msg = PoseStamped()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        msg.pose.position.x = self.x
        msg.pose.position.y = self.y
        msg.pose.position.z = 0.0

        # Convert yaw (theta) to quaternion
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = math.sin(self.theta / 2.0)
        msg.pose.orientation.w = math.cos(self.theta / 2.0)


        self.pose_pub.publish(msg)


        self.get_logger().info(
            f"x={self.x:.2f}, y={self.y:.2f}, linear vel={self.v:.2f}, angular vel={self.w:.2f}"
        )


def main(args=None):

    rclpy.init(args=args)

    node = RobotSimulator()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

# ros2 run robot_sim simulator