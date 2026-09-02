import cv2
import os
import json

from backbone_model.apriltag import AprilTag
from backbone_model.datasets.synthetic_dataset.generate_synthetic_data import random_background

def extract_frames(video_path, output_folder):
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0

    prev_robot = False

    ground_truth = []

    while True:
        # Read the next frame
        success, frame = video.read()
        
        # If the frame was not successfully read, we reached the end of the video
        if not success:
            break

        if not prev_robot:
            gt = AprilTag.get_ground_truth(frame)
            if gt["class"]==1:
                ground_truth.append(gt)
                
                # Save the frame as a JPEG file
                frame_name = f"frame_{frame_count:04d}.jpg"
                frame_path = os.path.join(output_folder, frame_name)
                cv2.imwrite(frame_path, frame)
                
                frame_count += 1
                prev_robot = True
        else:
            ground_truth.append({
                 "center": (0,0),
                 "orientation": 0,
                 "class:": 0
            })

            frame = random_background("/Users/mooncyli/Desktop/BU_RISE/BU-RISE/backbone_model/datasets/synthetic_dataset/backgrounds")
            
            # Save the frame as a JPEG file
            frame_name = f"frame_{frame_count:04d}.jpg"
            frame_path = os.path.join(output_folder, frame_name)
            cv2.imwrite(frame_path, frame)
            
            frame_count += 1
            prev_robot = False

    with open("/workspace/backbone_model/dataset/video_dataset1/ground_truth.json", "w") as file:
            json.dump(ground_truth, file, indent = 4)

        # Release the video capture object
    video.release()
    print(f"Successfully extracted {frame_count} frames.")