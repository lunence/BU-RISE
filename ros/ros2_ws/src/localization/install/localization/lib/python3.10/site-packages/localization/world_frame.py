import numpy as np
import cv2

from . import config
from localization.camera import Camera
from localization.apriltag import AprilTag

class WorldFrame:
    def __init__(self, init_image = None):
        if init_image is None:
            # loop until apriltag in frame of camera
            camera = Camera()

            while True:
                frame = camera.get_frame()

                tags = AprilTag.get_apriltags(frame)
                if tags:
                    init_image = frame
                    break

        # get homography from first tag of the image (only one in image)
        tag = AprilTag.get_apriltags(init_image)
        self.H = tag[0].homography
        self.H_inv = np.linalg.inv(self.H)
        self.R = tag[0].pose_R
        self.T = tag[0].pose_t

    # input NORMALIZED coords
    def pixel_to_world(self, pose):
        # x = pose[0]
        # y = pose[1]

        # if 0<=x<=1 and 0<=y<=1:
        #     x *= config.APRILTAG_WIDTH
        #     y *= config.APRILTAG_HEIGHT
        
        # p = np.array([x, y, 1.0])
    
        # world = self.H_inv @ p
        # world /= world[2]
    
        # return world[:2] * config.TAG_SIZE/2

        scale = np.array([config.APRILTAG_WIDTH, config.APRILTAG_HEIGHT])

        pose *= scale

        # Pixel coordinates
        pixel = np.array([[pose[0]],
                        [pose[1]],
                        [1.0]])

        # Ray in camera coordinates
        ray_cam = config.K_inverse @ pixel

        # Camera center in world coordinates
        camera_center = -self.R.T @ self.T

        # Ray direction in world coordinates
        ray_world = self.R.T @ ray_cam

        # Intersect the ray with the ground plane z = 0
        s = -camera_center[2, 0] / ray_world[2, 0]

        world = camera_center + s * ray_world

        return world[:2].flatten()

    def world_to_pixel(self, pose):
        # x = pose[0]
        # y = pose[1]
        
        # p = np.array([x, y, 1.0])
    
        # pixel = self.H @ p
        # pixel /= pixel[2]
    
        # return pixel[:2] / (config.TAG_SIZE/2) 

        pose = np.asarray(pose).flatten()

        # World point (z = 0 on the ground)
        world = np.array([[pose[0]],
                        [pose[1]],
                        [0.0]])

        # Transform to camera coordinates
        camera = self.R @ world + self.T

        # Project into image
        pixel = config.K @ camera
        pixel /= pixel[2]

        return pixel[:2].flatten()      

    def plot_homography_grid(self, image):
        spacing = 0.25
        extent = 2.0

        image = Camera.apriltag_resize(image)

        # Draw vertical world lines (constant X)
        for X in np.arange(-extent, extent + spacing, spacing):
            prev_pixel = None

            for Y in np.arange(-extent, extent + spacing, spacing):
                p_world = np.array([X, Y])

                p_pixel = self.world_to_pixel(p_world)
                p_pixel = p_pixel.astype(int)

                if prev_pixel is not None:
                    if (
                        0 <= p_pixel[0] < image.shape[1]
                        and 0 <= p_pixel[1] < image.shape[0]
                        and 0 <= prev_pixel[0] < image.shape[1]
                        and 0 <= prev_pixel[1] < image.shape[0]
                    ):
                        cv2.line(
                            image,
                            tuple(prev_pixel),
                            tuple(p_pixel),
                            (0, 255, 0),
                            1
                        )

                prev_pixel = p_pixel


        # Draw horizontal world lines (constant Y)
        for Y in np.arange(-extent, extent + spacing, spacing):
            prev_pixel = None

            for X in np.arange(-extent, extent + spacing, spacing):
                p_world = np.array([X, Y])

                p_pixel = self.world_to_pixel(p_world)
                p_pixel = p_pixel.astype(int)

                if prev_pixel is not None:
                    if (
                        0 <= p_pixel[0] < image.shape[1]
                        and 0 <= p_pixel[1] < image.shape[0]
                        and 0 <= prev_pixel[0] < image.shape[1]
                        and 0 <= prev_pixel[1] < image.shape[0]
                    ):
                        cv2.line(
                            image,
                            tuple(prev_pixel),
                            tuple(p_pixel),
                            (0, 255, 0),
                            1
                        )

                prev_pixel = p_pixel


        # Mark origin
        origin = self.world_to_pixel(np.array([0, 0])).astype(int)

        if (
            0 <= origin[0] < image.shape[1]
            and 0 <= origin[1] < image.shape[0]
        ):
            cv2.circle(image, tuple(origin), 5, (255, 0, 0), -1)

            cv2.putText(
                image,
                "(0,0)",
                tuple(origin + np.array([5, -5])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255,0,0),
                1
            )

        return image

    def plot_world_coords(self):
        return