import cv2
import torch

import model_training.util.config as config
from model_training.util.utils import crop_to_ratio
from model_training.model.transforms import get_transforms

class Camera:
    def __init__(self):
        url = "http://host.docker.internal:8080/video"

        self.cap = cv2.VideoCapture(url)

        print("Camera opened:", self.cap.isOpened())

        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera stream")

        # self.cap = cv2.VideoCapture("http://host.docker.internal:8080/video")
        # if not self.cap.isOpened():
        #     print("Failed to open camera")
        #     # exit()

    def get_frame(self):
        ret, frame = self.cap.read()

        if not ret:
            return None

        #convert frame to RGB and normalize pixel values to [0, 1] to match pytorch format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb

    @staticmethod
    def apriltag_resize(frame):
        # undistorted = cv2.undistort(frame, config.K, config.D)
        # reseize to 640x480
        cropped = crop_to_ratio(frame)
        resized = cv2.resize(cropped, (config.APRILTAG_WIDTH, config.APRILTAG_HEIGHT), interpolation=cv2.INTER_CUBIC)
        return resized

    @staticmethod
    def model_resize(frame):
        # undistorted = cv2.undistort(frame, config.K, config.D)
        # resize to 160x120
        cropped = crop_to_ratio(frame)
        resized = cv2.resize(cropped, (config.WIDTH, config.HEIGHT), interpolation=cv2.INTER_CUBIC)
        return resized

    @staticmethod
    # prep for model
    def model_prep(frame, device):
        resized = Camera.model_resize(frame)
        
        # image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # apply transformations
        transforms = get_transforms()
        image = transforms(resized)

        image = image.to(device)
        image = image.unsqueeze(0)

        return image

