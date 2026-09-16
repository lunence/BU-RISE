import cv2
from localization.camera import Camera

print("done importing")

camera = Camera()

while True:
    frame = camera.get_frame()

    Camera.send_stream(frame)
