import cv2
import numpy as np
import math

from localization.detector import Detector
from localization.camera import Camera
from localization import config
from localization.utils import normalize_coords


weights = "/workspace/ros2_ws/src/localization/localization/saved_models/best_finetuning_model_lr1e-3.pth"
init_image_path = "/workspace/ros2_ws/src/localization/localization/init_image/initialization_apriltag.jpg"


detector = Detector(weights, init_image_path)


def draw_pose(frame, center, orientation):
    """
    Draw predicted robot center and heading.
    Assumes:
        0 deg = +Y
        90 deg = +X
    """

    # convert world position back to pixel
    pixel = detector.worldframe.world_to_pixel(center)

    pixel = normalize_coords(
        pixel,
        config.APRILTAG_WIDTH,
        config.APRILTAG_HEIGHT,
        config.WIDTH,
        config.HEIGHT
    )

    cx, cy = pixel.astype(int)

    # draw center
    cv2.circle(
        frame,
        (cx, cy),
        2,
        (0, 0, 255),
        -1
    )

    # draw heading arrow
    length = 15
    theta = math.radians(orientation)

    end_x = int(cx + length * math.sin(theta))
    end_y = int(cy - length * math.cos(theta))

    cv2.arrowedLine(
        frame,
        (cx, cy),
        (end_x, end_y),
        (0, 255, 0),
        1,
        tipLength=0.3
    )

    cv2.putText(
        frame,
        f"{orientation:.0f}",
        (cx, cy - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3,
        (255,255,255),
        1
    )

    return frame


def main():

    camera = Camera()

    while True:

        frame = camera.get_frame()

        if frame is None:
            continue

        # model prediction
        center, orientation, robot_class = detector.predict_pose(frame)

        print(
            "center:",
            center,
            "orientation:",
            orientation,
            "class:",
            robot_class
        )

        # visualize
        frame = draw_pose(
            frame,
            center,
            orientation
        )

        # display at 160x120
        small = cv2.resize(
            frame,
            (160,120),
            interpolation=cv2.INTER_AREA
        )

        Camera.send_stream(small)



if __name__ == "__main__":
    main()