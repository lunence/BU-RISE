import numpy as np
import cv2

from . import config
from localization.camera import Camera
from localization.apriltag import AprilTag

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

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

        pose = np.asarray(pose, dtype=float).copy()
        # pose[1] = 1.0 - pose[1]
        pixel_xy = pose * scale

        # Pixel coordinates
        pixel = np.array([[pixel_xy[0]],
                        [pixel_xy[1]],
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
        world[1] *= -1

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
                        [-pose[1]],
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

    def visualize_frames(self, axis_length=0.5, ground_extent=2.0):
        """
        Visualize the world frame and camera frame in 3D.
        """

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

        # -------------------------------------------------
        # World frame
        # -------------------------------------------------

        origin = np.zeros(3)

        ax.quiver(*origin, axis_length, 0, 0,
                color="r", linewidth=2)
        ax.quiver(*origin, 0, axis_length, 0,
                color="g", linewidth=2)
        ax.quiver(*origin, 0, 0, axis_length,
                color="b", linewidth=2)

        ax.text(axis_length, 0, 0, "Xw")
        ax.text(0, axis_length, 0, "Yw")
        ax.text(0, 0, axis_length, "Zw")

        # -------------------------------------------------
        # Camera frame
        # -------------------------------------------------

        R_wc = self.R.T

        camera_center = (-R_wc @ self.T).flatten()

        # make world Z point upward
        camera_center[2] = -camera_center[2]

        print("Camera center:", camera_center)
        print("camera x:", R_wc[:,0])
        print("camera y:", R_wc[:,1])
        print("camera z:", R_wc[:,2])
        print("World Z in camera frame:", self.R[:,2])
        print(self.T)

        cam_x = R_wc[:, 0] * axis_length
        cam_y = -R_wc[:, 1] * axis_length
        cam_z = -R_wc[:, 2] * axis_length

        ax.quiver(*camera_center, *cam_x,
                color="r", linestyle="--")

        ax.quiver(*camera_center, *cam_y,
                color="g", linestyle="--")

        ax.quiver(*camera_center, *cam_z,
                color="b", linestyle="--")

        ax.text(*(camera_center + cam_x), "Xc")
        ax.text(*(camera_center + cam_y), "Yc")
        ax.text(*(camera_center + cam_z), "Zc")

        # -------------------------------------------------
        # Ground plane
        # -------------------------------------------------

        xx, yy = np.meshgrid(
            np.linspace(-ground_extent, ground_extent, 2),
            np.linspace(-ground_extent, ground_extent, 2)
        )

        zz = np.zeros_like(xx)

        ax.plot_surface(
            xx,
            yy,
            zz,
            alpha=0.2
        )

        # -------------------------------------------------
        # Ground grid
        # -------------------------------------------------

        spacing = 0.25

        for x in np.arange(-ground_extent, ground_extent + spacing, spacing):
            ax.plot(
                [x, x],
                [-ground_extent, ground_extent],
                [0, 0],
                color="gray",
                linewidth=0.5
            )

        for y in np.arange(-ground_extent, ground_extent + spacing, spacing):
            ax.plot(
                [-ground_extent, ground_extent],
                [y, y],
                [0, 0],
                color="gray",
                linewidth=0.5
            )

        # -------------------------------------------------
        # Camera center
        # -------------------------------------------------

        ax.scatter(*camera_center,
                color="k",
                s=40)

        ax.text(*camera_center, "Camera")

        # -------------------------------------------------

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")

        ax.set_box_aspect((1, 1, 1))

        scale = ground_extent
        ax.set_xlim(-scale, scale)
        ax.set_ylim(-scale, scale)
        ax.set_zlim(-0.2, scale)

        plt.savefig("camera_frames.png", dpi=300, bbox_inches="tight")
        plt.close(fig)



    
    def debug_origin_conversion(self):
        print("\n========== Coordinate Debug ==========")

        # World origin
        world_origin = np.array([0.0, 0.0])

        print("World coordinate:")
        print(world_origin)

        # World -> Pixel
        pixel = self.world_to_pixel(world_origin)

        print("\nPixel coordinate:")
        print(pixel)

        # Pixel -> World
        # pixel_to_world expects normalized coordinates,
        # so convert pixel back to normalized first
        normalized = np.array([
            pixel[0] / config.APRILTAG_WIDTH,
            pixel[1] / config.APRILTAG_HEIGHT
        ])

        print("\nNormalized pixel coordinate:")
        print(normalized)

        world_back = self.pixel_to_world(normalized)

        print("\nRecovered world coordinate:")
        print(world_back)

        print("\nError:")
        print(world_back - world_origin)

        print("=====================================\n")