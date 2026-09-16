def crop_to_ratio(image):
    height, width = image.shape[:2]
    crop_width = height * 4 // 3
    x_start = (width - crop_width) // 2   # 160
    x_end = x_start + crop_width          # 1120

    cropped = image[:, x_start:x_end]
    return cropped


from . import config
import numpy as np

def normalize_coords(pose, start_width, start_height, end_width, end_height, neg_coords = False):
    x = pose[0]
    y = pose[1]
    if(neg_coords):
        x += start_width/2
        y += start_height/2

    new_pose = [x/start_width*end_width, y/start_height*end_height]
    return new_pose

def normalized_to_model(pose):
    scale = np.array([config.WIDTH, config.HEIGHT])
    return pose * scale

def normalized_to_apriltag(pose):
    scale = np.array([config.APRILTAG_WIDTH, config.APRILTAG_HEIGHT])
    return pose * scale

import os
import re

def get_sorted_files(sequence_folder):
    image_files = []
    for file_name in os.listdir(sequence_folder):
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_files.append(os.path.join(sequence_folder, file_name))
    if not image_files:
        print("No images were found.")
        return []

    image_files = sorted(
        image_files,
        key=lambda filename: int(
            re.search(r"(\d+)(?=\.[^.]+$)", filename).group(1)
        ),
    )
    return image_files