import csv
import numpy as np
from geometry_msgs.msg import Twist
import os
from datetime import datetime

M = [
        [-0.5, 0],
        [-0.25, 0.5],
        [0, 0.25],
        [0.25, 0.50],
        [0.5, 0],
    ]

circle = [
        [-0.25, 0],
        [0, 0.25],
        [0.25, 0],
        [0, -0.25],
        [-0.25, 0],
    ]

idk = [
        [0,0],
        [-0.5, -0.5],
    ]

path = idk


class NavigationTest:

    def __init__(self, node):
        self.node = node

        # Add as many test points as you like
        self.points = path

        self.current = -1
        self.results = []

    def start(self):
        """Start the first test."""
        self.current = -1
        self.next_goal()

    def next_goal(self):

        self.current += 1

        if self.current >= len(self.points):
            self.finish()
            return

        goal = self.points[self.current]

        print()
        print("=" * 50)
        print(f"Starting Test {self.current + 1}")
        print("Goal:", goal)
        print("=" * 50)

        self.node.path = [goal]

        controller = self.node.controller

        controller.reached_target = False
        controller.exit = False
        controller.stall_counter = 0

    def goal_finished(self):

        if self.node.path is None:
            print("No active path!")
            self.next_goal()
            return

        goal = np.array(self.node.path[0])

        if self.node.gt_pose is None:
            error = None
        else:
            error = np.linalg.norm(goal - self.node.gt_pose)

        success = self.node.controller.reached_target

        self.results.append({
            "goal_x": goal[0],
            "goal_y": goal[1],
            "success": success,
            "error": error
        })

        print()
        print("Result")
        print("--------------------")
        print("Success:", success)
        print("Final error:", error)

        stop = Twist()
        self.node.cmd_publisher.publish(stop)

        input("\nPress ENTER for next waypoint...")

        self.next_goal()

    def finish(self):

        print()
        print("=" * 50)
        print("Navigation Testing Complete")
        print("=" * 50)

        print()
        notes = input(
            "Notes for this run (press Enter to skip): "
        ).strip()

        # Keep the CSV clean
        if notes == "":
            notes = ""

        filename = "navigation_results.csv"

        # Check if file already exists
        file_exists = os.path.isfile(filename)

        controller = self.node.controller


        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(filename, "a", newline="") as f:

            writer = csv.writer(f)

            # Only write header for a new file
            if not file_exists:
                writer.writerow([
                    "timestamp",
                    "goal_x",
                    "goal_y",
                    "success",
                    "error_m",
                    "success_radius",
                    "kp_linear",
                    "kp_angular",
                    "max_linear",
                    "max_angular",
                    "notes"
                ])

            for r in self.results:

                writer.writerow([
                    timestamp,
                    r["goal_x"],
                    r["goal_y"],
                    r["success"],
                    r["error"],
                    controller.success_radius,
                    controller.Kp_lin,
                    controller.Kp_turn,
                    controller.max_linear,
                    controller.max_angular,
                    notes
                ])

        print(f"Appended results to {filename}")

        successes = sum(r["success"] for r in self.results)

        print(f"Success rate: {successes}/{len(self.results)}")

        errors = [
            r["error"]
            for r in self.results
            if r["error"] is not None
        ]

        if len(errors) > 0:
            print("Average final error:", np.mean(errors))
            print("Maximum error:", np.max(errors))
            print("Minimum error:", np.min(errors))