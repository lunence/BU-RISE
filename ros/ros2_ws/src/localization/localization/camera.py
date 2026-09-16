import cv2
import torch
import requests
import numpy as np
import threading

from . import config
from localization.utils import crop_to_ratio
from localization.robot_detector_model.transforms import get_transforms

class Camera:
    def __init__(self):
        url = "http://host.docker.internal:8080/video"

        self.cap = cv2.VideoCapture(url)

        print("Camera opened:", self.cap.isOpened())

        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera stream")

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.frame = None
        self.running = True

        self.thread = threading.Thread(
            target=self.update_camera
        )
        self.thread.start()

        # self.cap = cv2.VideoCapture("http://host.docker.internal:8080/video")
        # if not self.cap.isOpened():
        #     print("Failed to open camera")
        #     # exit()

    def update_camera(self):
        while self.running:
            ret, frame = self.cap.read()

            if ret:
                # overwrite old frame
                self.frame = frame


    def get_frame(self):
        frame_rgb = self.frame
        if self.frame is not None:
            frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        return frame_rgb


    def shutdown(self):
        self.running = False
        self.thread.join()
        self.cap.release()

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

    @staticmethod
    def send_stream(frame):
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        _, jpg = cv2.imencode(".jpg", frame_bgr)

        requests.post(
            "http://host.docker.internal:8081/upload",
            data=jpg.tobytes(),
            headers={"Content-Type": "image/jpeg"},
        )