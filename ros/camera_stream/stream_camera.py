from flask import Flask, Response
import cv2
import atexit

app = Flask(__name__)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# # Optional: set resolution
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


def generate():
    print("Client connected")

    while True:
        success, frame = cap.read()

        if not success:
            print("Camera read failed")
            break

        ret, jpeg = cv2.imencode(".jpg", frame)

        if not ret:
            print("JPEG encode failed")
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + jpeg.tobytes()
            + b"\r\n"
        )



import signal
import sys

def shutdown(sig, frame):
    print("Shutting down...")
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown)

@atexit.register
def cleanup():
    print("Cleaning up camera...")
    cap.release()
    cv2.destroyAllWindows()

@app.route("/video")
def video():
    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, threaded=False)
    

# http://localhost:8080/video