import cv2
from localization.world_frame import WorldFrame


init_image = cv2.imread("/workspace/ros2_ws/src/localization/test_scripts/initialization_apriltag1.jpg")

world_frame = WorldFrame(init_image)

world_frame.visualize_frames()

