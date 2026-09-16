import math
import numpy as np


def predict_pose(pose, theta, v, omega, dt):
    # 0 degrees is +Y axis
    x_pred = pose[0] + v * math.sin(math.radians(theta)) * dt
    y_pred = pose[1] + v * math.cos(math.radians(theta)) * dt

    theta_pred = (theta - math.degrees(omega) * dt) % 360

    return np.array([x_pred, y_pred]), theta_pred


def test_predict_pose():

    tests = [
        {
            "name": "Move straight +Y",
            "pose": np.array([0.0, 0.0]),
            "heading": 0,
            "speed": 1.0,
            "omega": 0.0,
            "dt": 1.0,
            "expected_pose": [0.0, 1.0],
            "expected_heading": 0
        },

        {
            "name": "Move straight +X",
            "pose": np.array([0.0, 0.0]),
            "heading": 90,
            "speed": 1.0,
            "omega": -math.pi/2,
            "dt": 1.0,
            "expected_pose": [1.0, 0.0],
            "expected_heading": 180
        },

        {
            "name": "Rotate clockwise 90 degrees",
            "pose": np.array([0.0, 0.0]),
            "heading": 0,
            "speed": 0.0,
            "omega": math.pi/2,   # rad/s
            "dt": 1.0,
            "expected_pose": [0.0, 0.0],
            "expected_heading": 270
        },

        {
            "name": "Move +Y while rotating",
            "pose": np.array([1.0, 2.0]),
            "heading": 0,
            "speed": 2.0,
            "omega": math.pi/2,
            "dt": 1.0,
            "expected_pose": [1.0, 4.0],
            "expected_heading": 270
        }
    ]


    for test in tests:
        result_pose, result_heading = predict_pose(
            test["pose"],
            test["heading"],
            test["speed"],
            test["omega"],
            test["dt"]
        )

        print("\n----------------------")
        print(test["name"])

        print("Result pose:", result_pose)
        print("Expected pose:", test["expected_pose"])

        print("Result heading:", result_heading)
        print("Expected heading:", test["expected_heading"])


if __name__ == "__main__":
    test_predict_pose()