import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import math
import cv2
import numpy as np
from std_srvs.srv import Trigger
import matplotlib.pyplot as plt
import time

from localization.detector import Detector
from localization.camera import Camera
from localization.trajectory_plot import TrajectoryPlot

class LocalizationNode(Node):
    def __init__(self, model_path, init_image_path):
        super().__init__("localization")
        self.detector = Detector(model_path, init_image_path)
        self.camera = Camera()
        self.trajectory = TrajectoryPlot()

        self.last_center = None
        self.last_orientation = None
        self.last_capture_time = None

        # create publisher for model
        self.model_pose_publisher = self.create_publisher(
            PoseStamped,
            "/pred_pose",
            10
        )

         # create publisher
        self.apriltag_pose_publisher = self.create_publisher(
            PoseStamped,
            "/ground_truth_pose",
            10
        )

        # # timer 
        # self.model_timer = self.create_timer(
        #     0.05,
        #     self.localization_callback
        # )


        self.pose_service = self.create_service(
            Trigger,
            "/get_current_pose",
            self.get_current_pose
        )

        self.latency_history = []
        self.frame_history = []
        self.frame_count = 0

        self.start_time = time.time()

        self.position_error_history = []
        self.orientation_error_history = []
        self.time_history = []

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

    def localization_callback(self):
        gt_center = np.array([0,0])
        gt_orientation = 0
        gt_class = 0

        start = self.get_clock().now()
        frame = self.camera.get_frame()
        end = self.get_clock().now()

        camera_read_ms = (end - start).nanoseconds / 1e6
        print(f"Camera read time: {camera_read_ms:.1f} ms")

        if frame is None:
            return

        capture_time = self.get_clock().now()
        
        pred_center, pred_orientation, pred_class = self.detector.predict_pose(frame)

        # position_changed = (
        #     self.last_center is None or
        #     np.linalg.norm(pred_center - self.last_center) > 0.02
        # )

        # heading_changed = (
        #     self.last_orientation is None or
        #     abs(((pred_orientation - self.last_orientation + 180) % 360) - 180) > 5
        # )

        # if not (position_changed or heading_changed):
        #     print("no change in position or heading\n")
        #     return
        #     capture_time = self.last_capture_time

        gt_center, gt_orientation, gt_class = self.detector.ground_truth_pose(frame)

        print("Pred class:", pred_class)
        print("GT class:", gt_class)

        # resize  to be same res as what is sent into the model
        frame = Camera.model_resize(frame)

        if pred_class == 1:
            self.publish_pose(
                pred_center,
                pred_orientation,
                capture_time,
                self.model_pose_publisher
            )

            self.trajectory.add_prediction(pred_center)

            self.last_center = pred_center.copy()
            self.last_orientation = pred_orientation
            self.last_capture_time = capture_time

            # plot predictions (red)
            frame = self.detector.plot(frame, pred_center, pred_orientation, (0,0,255))
            
        
        if gt_class == 1:
            self.publish_pose(
                gt_center,
                gt_orientation,
                capture_time,
                self.apriltag_pose_publisher
            )

            self.trajectory.add_ground_truth(gt_center)

            # plot gt (green)
            frame = self.detector.plot(frame, gt_center, gt_orientation, (0,255,0))
            
        Detector.print_all(pred_center, pred_orientation, pred_class, gt_center, gt_orientation, gt_class) 

        if pred_class == 1 and gt_class == 1:
            position_error, orientation_error = Detector.calculate_errors(
                pred_center,
                pred_orientation,
                gt_center,
                gt_orientation
            )

            elapsed = time.time() - self.start_time

            self.position_error_history.append(position_error)
            self.orientation_error_history.append(orientation_error)
            self.time_history.append(elapsed)           

        Camera.send_stream(frame)

        self.record_latency(start)

        print()

        
    def publish_pose(self, center, orientation, capture_time, publisher):
        # convert orientation to format for PoseStamped (only change yaw)
        msg = PoseStamped()

        msg.header.stamp = capture_time.to_msg()
        msg.header.frame_id = "map"

        # add center to message
        msg.pose.position.x = center[0]
        msg.pose.position.y = center[1]
        msg.pose.position.z = 0.0

        ros_yaw = 90 - orientation
        ros_yaw = ros_yaw % 360

        yaw_rad = math.radians(ros_yaw)

        # add orientation to message (convert to radians first for trig)
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = math.sin((yaw_rad) / 2.0)
        msg.pose.orientation.w = math.cos((yaw_rad) / 2.0)

        publisher.publish(msg)

    def get_current_pose(self, request, response):
        if self.last_center is None:
            response.success = False
            response.message = "No pose available"
            return response

        msg = PoseStamped()

        msg.header.stamp = self.last_capture_time.to_msg()
        msg.header.frame_id = "map"

        msg.pose.position.x = float(self.last_center[0])
        msg.pose.position.y = float(self.last_center[1])
        msg.pose.position.z = 0.0

        yaw = math.radians(90 - self.last_orientation)

        msg.pose.orientation.z = math.sin(yaw / 2)
        msg.pose.orientation.w = math.cos(yaw / 2)

        self.model_pose_publisher.publish(msg)

        response.success = True
        response.message = "Published current pose"

        return response

    def record_latency(self, start_time):
        """
        Record end-to-end latency of one localization iteration and
        continuously update a latency graph.
        """
        

        end_time = self.get_clock().now()
        latency_ms = (end_time - start_time).nanoseconds / 1e6

        self.frame_count += 1
        self.frame_history.append(self.frame_count)
        self.latency_history.append(latency_ms)

        # Keep only the latest 200 frames
        if len(self.latency_history) > 200:
            self.latency_history.pop(0)
            self.frame_history.pop(0)

        print(f"Total latency: {latency_ms:.1f} ms")

    def plot_latency(self):
        plt.figure(figsize=(8, 4))
        plt.plot(self.frame_history, self.latency_history)
        plt.xlabel("Frame")
        plt.ylabel("Latency (ms)")
        plt.title("Localization Pipeline Latency")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("latency_graph.png")
        plt.close()

    def plot_error_vs_time(self):
        if len(self.time_history) == 0:
            print("No error data collected.")
            return

        plt.figure(figsize=(10, 6))

        plt.plot(
            self.time_history,
            self.position_error_history,
            label="Position Error (m)",
            linewidth=2
        )

        plt.plot(
            self.time_history,
            self.orientation_error_history,
            label="Orientation Error (deg)",
            linewidth=2
        )

        plt.xlabel("Time (s)")
        plt.ylabel("Error")
        plt.title("Localization Error vs Time")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("errors_graph.png")
        plt.close()

        print(f"Average position error: {np.mean(self.position_error_history):.3f} m")
        print(f"Average orientation error: {np.mean(self.orientation_error_history):.2f}°")

def main():
    import rclpy

    saved_model = "/workspace/ros2_ws/src/localization/localization/saved_models/best_finetuning_model_new_synthetic.pth"
    init_image_path = "/workspace/ros2_ws/src/localization/localization/init_image/initialization_apriltag.jpg"

    rclpy.init()
    node = LocalizationNode(saved_model, init_image_path)
    try:
        while rclpy.ok():
            node.localization_callback()
            rclpy.spin_once(node, timeout_sec=0)

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        print("Saving graphs...")

        node.plot_latency()
        node.plot_error_vs_time()
        node.trajectory.plot()

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()

# colcon build
# source install/setup.bash
# ros2 run localization localization_node