import cv2
import numpy as np

from localization.world_frame import WorldFrame
from localization.apriltag import AprilTag

from localization.detector import Detector
from localization.camera import Camera
from localization.utils import normalize_coords, crop_to_ratio
from localization import config


weights = "/workspace/ros2_ws/src/localization/localization/saved_models/best_finetuning_model_lr1e-3.pth"
init_image_path = "/workspace/ros2_ws/src/localization/localization/init_image/initialization_apriltag.jpg"


detector = Detector(weights, init_image_path)

frame = cv2.imread(
    "/workspace/ros2_ws/src/localization/test_scripts/limo/limo1.jpg"
)


def test_pixel_reprojection(detector, frame):
    scale = [160,120]
    logits = detector.model.predict(frame)

    raw_pixel = logits["center"].numpy().flatten()

    print("\n========== COORDINATE PIPELINE ==========")
    print("Raw model pixel:")
    print(raw_pixel*scale)

    # -----------------------------
    # Pixel -> World
    # -----------------------------
    world = detector.worldframe.pixel_to_world(raw_pixel)

    print("\nAfter pixel_to_world():")
    print(world)

    # -----------------------------
    # World -> Pixel
    # -----------------------------
    pixel = detector.worldframe.world_to_pixel(world)

    print("\nAfter world_to_pixel():")
    print(pixel)

    # -----------------------------
    # Normalize for plotting
    # -----------------------------
    pixel_norm = normalize_coords(
        pixel,
        config.APRILTAG_WIDTH,
        config.APRILTAG_HEIGHT,
        config.WIDTH,
        config.HEIGHT,
    )

    print("\nAfter normalize_coords():")
    print(pixel_norm)

    # -----------------------------
    # Errors
    # -----------------------------
    diff = pixel_norm - raw_pixel*scale

    print("\nDifference (reconstructed - raw):")
    print(diff)

    print("\nAbsolute difference:")
    print(np.abs(diff))

    print("\nL2 error:")
    print(np.linalg.norm(diff))

    # -----------------------------
    # Draw everything
    # -----------------------------
    display = frame.copy()
    display = crop_to_ratio(display)
    display = Camera.model_resize(display)

    # Raw model prediction (red)
    x, y = (raw_pixel*scale).astype(int)
    cv2.circle(display, (x, y), 5, (0, 0, 255), -1)

    # Reprojected pixel (green)
    x2, y2 = pixel_norm.astype(int)
    cv2.circle(display, (x2, y2), 5, (0, 255, 0), -1)

    # Line connecting them
    cv2.line(display, (x, y), (x2, y2), (255, 255, 0), 2)

    cv2.imwrite("/workspace/ros2_ws/src/localization/test_scripts/limo1_plotted.png", display)



test_pixel_reprojection(detector, frame)

# init_image = cv2.imread("/workspace/ros2_ws/src/localization/test_scripts/initialization_apriltag1.jpg")

# world_frame = WorldFrame(init_image)

# init_ground_truth = AprilTag.get_ground_truth(init_image)
# print(init_ground_truth)

# image = world_frame.plot_homography_grid(init_image)
# cv2.imwrite("/workspace/ros2_ws/src/localization/test_scripts/homography_grid.png", image)

# robot_image = cv2.imread("/workspace/ros2_ws/src/localization/test_scripts/robot_test.jpg")
# robot_ground_truth = AprilTag.get_ground_truth(robot_image)
# print(robot_ground_truth)

# center = init_ground_truth["center"]
# print(center)

# print(
#     world_frame.world_to_pixel(np.array([0.0, 0.0]))
# )

# cd /workspace/ros2_ws
# export PYTHONPATH=$PWD/src/localization:$PYTHONPATH
# python3 src/localization/test_scripts/test_world_transformations.py