import cv2
import os
import json

from backbone_model.util.apriltag import AprilTag
from backbone_model.datasets.synthetic_dataset.generate_synthetic_data import random_background


def extract_frames(video_path, output_folder):
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Error: Could not open video.")
        return

    os.makedirs(output_folder, exist_ok=True)

    frame_count = 0
    video_frame_count = 0

    # Process every 5th frame
    frame_skip = 15

    ground_truth = []

    background_folder = (
        "/Users/mooncyli/Desktop/BU_RISE/BU-RISE/"
        "backbone_model/datasets/synthetic_dataset/backgrounds"
    )

    while True:
        # Read the next video frame
        success, frame = video.read()

        if not success:
            break

        video_frame_count += 1

        # Only process every 5th frame
        if video_frame_count % frame_skip != 0:
            continue

        # Try to detect the robot using AprilTags
        gt = AprilTag.get_ground_truth(frame)

        if gt["class"] == 1:
            # ------------------------------------------------
            # Robot detected
            # ------------------------------------------------

            ground_truth.append(gt)

            frame_name = f"frame_{frame_count:04d}.jpg"
            frame_path = os.path.join(output_folder, frame_name)

            cv2.imwrite(frame_path, frame)

            print(
                f"Saved robot frame {frame_count}: "
                f"center={gt['center']}, "
                f"orientation={gt['orientation']:.2f}"
            )

        else:
            # ------------------------------------------------
            # Robot not detected -> save random background
            # ------------------------------------------------

            background_gt = {
                "center": (0, 0),
                "orientation": 0,
                "class": 0
            }

            ground_truth.append(background_gt)

            # Generate a random background
            background = random_background(background_folder)

            frame_name = f"frame_{frame_count:04d}.jpg"
            frame_path = os.path.join(output_folder, frame_name)

            cv2.imwrite(frame_path, background)

            print(
                f"Saved background frame {frame_count}"
            )

        frame_count += 1

    # Save ground truth
    ground_truth_path = os.path.join(
        output_folder,
        "ground_truth.json"
    )

    with open(ground_truth_path, "w") as file:
        json.dump(ground_truth, file, indent=4)

    video.release()

    print("\nFinished!")
    print(f"Read {video_frame_count} video frames.")
    print(f"Saved {frame_count} dataset images.")
    print(f"Ground truth saved to: {ground_truth_path}")


def main():
    extract_frames(
        "backbone_model/datasets/video_dataset2/video_dataset2.mov",
        "backbone_model/datasets/video_dataset2/frames"
    )


if __name__ == "__main__":
    main()
