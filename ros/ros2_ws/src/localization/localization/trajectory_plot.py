import matplotlib.pyplot as plt
import numpy as np


class TrajectoryPlot:
    def __init__(self):
        self.pred_x = []
        self.pred_y = []

        self.gt_x = []
        self.gt_y = []

    def add_prediction(self, pose):
        self.pred_x.append(pose[0])
        self.pred_y.append(pose[1])

    def add_ground_truth(self, pose):
        self.gt_x.append(pose[0])
        self.gt_y.append(pose[1])

    def plot(self):
        plt.figure(figsize=(7,7))

        plt.plot(
            self.gt_x,
            self.gt_y,
            'g-',
            linewidth=2,
            label="AprilTag Ground Truth"
        )

        plt.plot(
            self.pred_x,
            self.pred_y,
            'r--',
            linewidth=2,
            label="Model Prediction"
        )

        # show start
        if len(self.gt_x) > 0:
            plt.scatter(
                self.gt_x[0],
                self.gt_y[0],
                marker='o',
                s=100,
                label="Start"
            )

        # show end
        if len(self.gt_x) > 0:
            plt.scatter(
                self.gt_x[-1],
                self.gt_y[-1],
                marker='x',
                s=100,
                label="Finish"
            )

        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.title("Robot Trajectory")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()

        plt.show()