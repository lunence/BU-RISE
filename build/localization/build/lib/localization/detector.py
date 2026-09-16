import cv2
import torch
import math
import numpy as np
import time

from . import config
from localization.model import Model
from localization.apriltag import AprilTag
from localization.world_frame import WorldFrame
from localization.utils import normalize_coords

"""
model's predicted poses + homography to return real world coords
"""

class Detector:
    prev_ground_truth = {}

    def __init__(self, model_path, init_image_path = None):
        self.model = Model(model_path)
        self.apriltag = AprilTag()
        if init_image_path is not None:
            init_image = cv2.imread(init_image_path)
        else:
            init_image = None
        self.worldframe = WorldFrame(init_image)

        self.last_orientation = 0

    def predict_pose(self, frame):
        start = time.perf_counter()

        logits = self.model.predict(frame)

        inference_time = time.perf_counter() - start
        print(f"Model latency: {inference_time:.3f} s")

        # center pred (normalized)
        pred_center = logits["center"].numpy().flatten()
        print("normalized center:", pred_center)

        # orientation pred (degrees)
        orientation_probs = torch.softmax(logits["orientation"], dim=1)

        confidence, orientation_bin = torch.max(orientation_probs, dim=1)

        pred_orientation = orientation_bin.item() * 5
        orientation_confidence = confidence.item()

        print(
            "Orientation:",
            pred_orientation,
            "Confidence:",
            orientation_confidence
        )

        if orientation_confidence < 0.4:
            print("Low orientation confidence, ignoring")
            pred_orientation = self.last_orientation
        else:
            self.last_orientation = pred_orientation

        pred_class = logits["class"].argmax(dim=1).item()

        pred_center_world = self.worldframe.pixel_to_world(pred_center)

        return pred_center_world, pred_orientation, pred_class
    
    def ground_truth_pose(self, frame):
        ground_truth = self.apriltag.get_ground_truth(frame)

        # if ground_truth["class"] == 0:
        #     ground_truth = Detector.prev_ground_truth

        gt_center = np.array(ground_truth["center"]).flatten()
        gt_orientation = ground_truth["orientation"]
        gt_class = ground_truth["class"]

        gt_center_world = self.worldframe.pixel_to_world(gt_center)

        Detector.prev_ground_truth = ground_truth

        return gt_center_world, gt_orientation, gt_class
    
    @staticmethod
    def calculate_errors(pred_center, pred_orientation, gt_center, gt_orientation):
        orientation_error = abs(pred_orientation - gt_orientation)
        orientation_error = min(orientation_error, 360 - orientation_error)

        center_error = np.sqrt((pred_center[1]-gt_center[1])**2 + (pred_center[0]-gt_center[0])**2)

        return center_error, orientation_error

    #TODO: modify so it can be used for gt and predictions independently
    @staticmethod
    def print_all(pred_center, pred_orientation, pred_class, gt_center, gt_orientation, gt_class):
        center_error, orientation_error = Detector.calculate_errors(pred_center, pred_orientation, gt_center, gt_orientation)

        # Print ground truth
        print("\nGround Truth")
        print("------------")

        print("Centers:", gt_center)

        print("Orientations (angle):", gt_orientation)

        print("Class:", gt_class)

        print()

        # Print predictions
        print("Prediction")
        print("----------")

        print("Centers:", pred_center)

        print("Orientations (angle):", pred_orientation)

        print("Class:", pred_class)

        print()
        print("Center Error:", center_error)
        print("Orientation Error:", orientation_error)
        
    def plot(self, frame, center, orientation, color = (0,0,255)):
        center_world = self.worldframe.world_to_pixel(center)
        center_world = normalize_coords(center_world, config.APRILTAG_WIDTH, config.APRILTAG_HEIGHT, config.WIDTH, config.HEIGHT)

        #predicted center coords
        cx, cy = center_world
        # cy = config.HEIGHT - cy

        # draw line in direction of angle
        length = 20  # length of the arrow in pixels

        angle = orientation  # degrees
        theta = math.radians(angle)

        end_x = int(cx + length * math.sin(theta))
        end_y = int(cy - length * math.cos(theta))  # subtract because image y-axis points down

        cv2.line(
            frame,
            (int(cx), int(cy)),
            (end_x, end_y),
            color,   
            2
        )

        # show center point (red)
        cv2.circle(frame, (int(cx), int(cy)), radius=3, color=color, thickness=-1)

        pose = self.worldframe.world_to_pixel(np.array([0,0]))
        pose = normalize_coords(pose, config.APRILTAG_WIDTH, config.APRILTAG_HEIGHT, config.WIDTH, config.HEIGHT)
        x = pose[0]
        y = pose[1]
        cv2.circle(frame, (int(x), int(y)), radius=3, color=(255,0,0), thickness=-1)

        # cv2.putText(frame,
        #             f"({cx}, {cy}",
        #             (int(cx), int(cy-10)),
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             0.6,
        #             (0,0,255),
        #             2)

        from pure_pursuit.test_path import path

        goal_points = path

        # Draw goal points
        if goal_points is not None:
            for i, goal in enumerate(goal_points):
                p = self.worldframe.world_to_pixel(np.array(goal))
                print("unnormalized")
                p = normalize_coords(
                    p,
                    config.APRILTAG_WIDTH,
                    config.APRILTAG_HEIGHT,
                    config.WIDTH,
                    config.HEIGHT,
                )

                gx, gy = p
                # gy = config.HEIGHT - gy

                cv2.circle(frame, (int(gx), int(gy)), 2, (255, 0, 255), -1)

                cv2.putText(
                    frame,
                    f"({gx}, {gy})",
                    (int(gx) + 3, int(gy) - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.25,           # small for 160x120 images
                    (255, 0, 255),
                    1,
                    cv2.LINE_AA,
                )


        return frame