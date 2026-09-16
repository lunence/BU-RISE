import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
import math
import numpy as np
from rclpy.time import Time
from std_srvs.srv import Trigger

from pure_pursuit.controller import PurePursuit
from pure_pursuit.test_path import NavigationTest

def ask_limo_number():
    limo_topic = "/limo"+input("Enter the last 3 numbers on the front of your LIMO: ").strip()+"/cmd_vel"
    return limo_topic

class PurePursuitNode(Node):
    def __init__(self, lookAheadDis, limo_topic, testing):
        super().__init__("pure_pursuit")
        self.controller = PurePursuit(lookAheadDis)

        self.testing = testing

        # for publishing outputs to limo
        self.cmd_publisher = self.create_publisher(
            Twist,
            limo_topic,
            10
        )

        # choose either to use apriltag or model poses
        # ros2 run pure_pursuit pure_pursuit_node --ros-args -p pose_source:=model
        # ros2 run pure_pursuit pure_pursuit_node --ros-args -p pose_source:=apriltag
        self.declare_parameter("pose_source", "model")
        pose_source = self.get_parameter(
            "pose_source"
        ).get_parameter_value().string_value

        if pose_source == "apriltag":
            topic = "/ground_truth_pose"
        elif pose_source == "model":
            topic = "/pred_pose"
        else:
            topic = "/robot_pose"

        self.pose_subscriber = self.create_subscription(
            PoseStamped,
            topic,
            self.pose_callback,
            10
        )

        self.gt_pose = None

        self.gt_subscriber = self.create_subscription(
            PoseStamped,
            "/ground_truth_pose",
            self.gt_callback,
            10
        )

        self.get_logger().info(f"Using pose source: {topic}")

        # single point for now
        self.path = None

        self.test = NavigationTest(self)

        # pure pursuit inputs
        self.current_pose = None
        self.current_heading = None
        self.speed = 0
        self.steering = 0
        self.measurement_time = None

        self.pred_pose = None
        self.pred_heading = None
        self.last_prediction_time = None

        self.waiting_for_final_gt = False
        self.gt_updates_after_stop = 0

        self.gt_measurement_time = None

        # timer 
        self.timer = self.create_timer(
            0.05,
            self.control_loop
        )

        self.pose_client = self.create_client(
            Trigger,
            "/get_current_pose"
        )

        while not self.pose_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for localization service...")

        self.request_current_pose()

    def request_current_pose(self):

        request = Trigger.Request()

        future = self.pose_client.call_async(request)

        future.add_done_callback(self.pose_response)


    def pose_response(self, future):

        response = future.result()

        if response.success:
            print("Localization pose requested successfully")
        else:
            print("No localization pose available")

    def control_loop(self):
        if self.current_pose is None or self.current_heading is None:
            print("No pose or heading")
            return

        if self.path is None:
            print("No path")
            if self.testing:
                self.test.goal_finished()
            else:
                self.get_new_goal()
            return

        if self.controller.reached_target or self.controller.exit:
            if self.controller.reached_target:
                print("\nTarget reached!")
                # if self.gt_pose is not None:
                #     target = np.array(self.path[0])

                #     real_error = np.linalg.norm(
                #         target - self.gt_pose
                #     )

                #     print("Target:", target)
                #     print("AprilTag position:", self.gt_pose)
                #     print("Actual final error:", real_error, "m")
            elif self.controller.exit:
                print("\nTarget not reached, exited early")

            # Stop the robot
            stop = Twist()
            self.cmd_publisher.publish(stop)

            self.stop_time = self.get_clock().now()
            self.waiting_for_final_gt = True

            if self.testing:
                self.test.goal_finished()
            else:
                self.get_new_goal()
            return

        if self.measurement_time is None:
            return

        print("actual position:", self.current_pose)
        print("actual heading:", self.current_heading)
        print()
        
        current_time = self.get_clock().now()

        dt = (
            current_time - self.last_prediction_time
        ).nanoseconds * 1e-9


        self.pred_pose, self.pred_heading = PurePursuitNode.predict_pose(
            self.pred_pose,
            self.pred_heading,
            self.speed,
            self.steering,
            dt
        )

        self.last_prediction_time = current_time

        self.speed, self.steering = self.controller.compute_control(
            self.path,
            self.pred_pose,
            self.pred_heading
        )

        cmd = Twist()

        cmd.linear.x = float(self.speed)
        cmd.angular.z = float(self.steering)

        self.cmd_publisher.publish(cmd)

        print("\nPure Pursuit")
        print("----------------")
        print("pose:", self.pred_pose)
        print("heading:", self.pred_heading)
        print("linear vel:", self.speed)
        print("angular vel:", self.steering)
        print(f"Measurement age: {dt:.3f} s")
        print()

    def pose_callback(self, msg):
        self.measurement_time = Time.from_msg(msg.header.stamp)

        self.current_pose = np.array([
            msg.pose.position.x,
            msg.pose.position.y
        ])

        q = msg.pose.orientation

        yaw = math.atan2(
            2.0 * (q.w*q.z + q.x*q.y),
            1.0 - 2.0*(q.y*q.y + q.z*q.z)
        )

        yaw_deg = math.degrees(yaw)

        self.current_heading = (90 - yaw_deg) % 360


        # reset prediction whenever a new measurement arrives
        self.pred_pose = self.current_pose.copy()
        self.pred_heading = self.current_heading
        self.last_prediction_time = self.get_clock().now()

    def gt_callback(self, msg):
        self.gt_pose = np.array([
            msg.pose.position.x,
            msg.pose.position.y
        ])

        self.gt_measurement_time = Time.from_msg(msg.header.stamp)

        if self.waiting_for_final_gt:

            # Ignore any image captured before stopping
            if self.gt_measurement_time <= self.stop_time:
                return

            age = (
                self.get_clock().now() - self.gt_measurement_time
            ).nanoseconds * 1e-9
            print("waiting...")

            if age < 1.0:
                target = np.array(self.path[0])

                real_error = np.linalg.norm(target - self.gt_pose)

                print("Target:", target)
                print("Final AprilTag:", self.gt_pose)
                print("Final error:", real_error)

                self.waiting_for_final_gt = False

    @staticmethod
    def predict_pose(pose, theta, v, omega, dt):
        # 0 degrees is pos y axis
        x_pred = pose[0] + v * math.sin(theta / 180 * math.pi) * dt
        y_pred = pose[1] + v * math.cos(theta / 180 * math.pi) * dt
        theta_pred = (theta - omega * 180 / math.pi * dt) % 360

        return np.array([x_pred, y_pred]), theta_pred

    # TODO: make it do this writes to the same file as the rest
    def get_new_goal(self):
        x = float(input("Goal x (m): "))
        y = float(input("Goal y (m): "))

        self.path = [[x, y]]

        # Reset controller
        self.controller.reached_target = False
        self.controller.exit = False

def main():
    import rclpy

    topic = ask_limo_number()

    try:
        lookAheadDis = 1

        rclpy.init()
        node = PurePursuitNode(lookAheadDis, topic, testing=True)
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received")  
    finally:
        # Always force a zero-velocity state on exit
        if node is not None:
            stop_twist = Twist()
            stop_twist.linear.x = 0.0
            stop_twist.angular.z = 0.0
            print("\nEmergency stop sent. Exiting cleanly.")
            node.cmd_publisher.publish(stop_twist)
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

        

if __name__ == "__main__":
    main()

# colcon build
# source install/setup.bash
# ros2 run pure_pursuit pure_pursuit_node