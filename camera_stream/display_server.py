from flask import Flask, request, Response
import cv2
import numpy as np

import os
from datetime import datetime

recording = False
video_writer = None

SAVE_DIR = "/Users/mooncyli/Desktop/BU_RISE/BU-RISE/videos"   # or any mounted volume
os.makedirs(SAVE_DIR, exist_ok=True)

app = Flask(__name__)

latest_frame = None


@app.route("/upload", methods=["POST"])
def upload():
    global latest_frame, recording, video_writer

    jpg = request.data

    img = cv2.imdecode(
        np.frombuffer(jpg, np.uint8),
        cv2.IMREAD_COLOR
    )

    print("Received frame", img.shape)

    latest_frame = img

    if recording:
        print("Writing frame")
        if video_writer is None:
            h, w = img.shape[:2]

            filename = datetime.now().strftime("%Y%m%d_%H%M%S.avi")
            path = os.path.join(SAVE_DIR, filename)

            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            video_writer = cv2.VideoWriter(path, fourcc, 30.0, (w, h))
            print("Opened:", video_writer.isOpened())

            print(f"Recording to {path}")

        video_writer.write(img)

    return "OK"

# curl http://localhost:8081/start_recording
@app.route("/start_recording")
def start_recording():
    global recording
    recording = True
    return "Recording started"

# curl http://localhost:8081/stop_recording
@app.route("/stop_recording")
def stop_recording():
    global recording, video_writer

    recording = False

    if video_writer is not None:
        video_writer.release()
        video_writer = None

    return "Recording stopped"


def generate():
    global latest_frame

    while True:
        if latest_frame is None:
            continue

        _, jpg = cv2.imencode(".jpg", latest_frame)

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + jpg.tobytes()
            + b"\r\n"
        )


@app.route("/video")
def video():
    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


app.run(host="0.0.0.0", port=8081)
# http://localhost:8081/video