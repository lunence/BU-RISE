import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
from model.model import GridNet
from model_training.datasets.objects.video_data_objects import device, dataset_test
from model_training.datasets.objects.real_world_objects import dataset_real_world
from model_training.util.config import WIDTH, HEIGHT
from model_training.util.world_frame import WorldFrame

def visualize(model_path):
    image = cv2.imread("/home/mooncyli/BU-RISE/model_training/initialization_apriltag.jpg")
    worldframe = WorldFrame(image)

    # idx = 0  # choose any sample
    model = GridNet().to(device)

    model.load_state_dict(torch.load(model_path,
                                    map_location=device))
    model.to(device)
    model.eval()

    for idx in range(20):
        # Get sample
        image, target = dataset_test[idx]

        # Skip samples with no valid ground truth
        if (
            torch.all(target["center"] == 0)
            and target["orientation"].item() == 0
        ):
            continue

        with torch.no_grad():
            images = image.unsqueeze(0).to(device)  # (1, 3, H, W)
            logits = model(images)

            scale = torch.tensor([WIDTH, HEIGHT], device=device)

            pred_center = logits["center"][0]
            pred_center *= scale
            pred_orientation = logits["orientation"][0].argmax()

            if target:
                gt_center = target["center"]
                gt_center *= scale
                gt_orientation = target["orientation"]


        # Convert image for plotting
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3)

        img = image.permute(1, 2, 0).cpu()
        img = img * std + mean          # undo normalization
        img = img.clamp(0, 1)
        img = (img.numpy() * 255).astype(np.uint8)

        # Print ground truth
        print("Ground Truth")
        print("------------")

        if "center" in target:
            print("Centers:", gt_center)

        if "orientation" in target:
            print("Orientations (bins):", gt_orientation)
            print("Orientations (angle):", gt_orientation*5)

        print()

        # Print predictions
        print("Prediction")
        print("----------")

        if "center" in logits:
            print("Centers:", pred_center)

        if "orientation" in logits:
            print("Orientations (bins):", pred_orientation)
            print("Orientations (angle):", pred_orientation*5)

        
        if target:
            orientation_error = torch.abs(pred_orientation - gt_orientation)
            orientation_error = torch.minimum(orientation_error*5, 360 - orientation_error*5)

            print()
            print("Center Error:", torch.norm(pred_center-gt_center))
            print("Orientation Error:", orientation_error*5)

        #TODO: make function for displaying center points
        cx, cy = pred_center.cpu().tolist()
        pred_world = worldframe.pixel_to_world([cx, cy])
        print("Predicted World Coords:", pred_world)

        # cv2.putText(img,
        #             f"{pred_orientation*5} deg",
        #             (int(cx), int(cy-10)),
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             0.6,
        #             (0,255,0),
        #             2)

        # show predicted center point (red)
        cv2.circle(img, (int(cx), int(cy)), radius=2, color=(0, 0, 255), thickness=-1)
        cv2.putText(img,
                    f"({cx}, {cy}",
                    (int(cx), int(cy-10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,0,255),
                    2)

        if target:
            # show actual center point (green)
            cx, cy = gt_center.cpu().tolist()

            gt_world = worldframe.pixel_to_world([cx, cy])
            print("Actual World Coords:", gt_world)

            cv2.circle(img, (int(cx), int(cy)), radius=2, color=(0, 255, 0), thickness=-1)
            cv2.putText(img,
                        f"({cx}, {cy}",
                        (int(cx), int(cy-10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0,255,0),
                        2)


        # Display image
        #cv2.imshow("Robot Detection", img_bgr)
        
        # adds grid to image
        # img = show_homography_grid(img)

        plt.figure("Image")
        plt.imshow(img, aspect="equal")
        plt.axis("off")
        # plt.show()

        # #show world coords
        # plt.figure("World Coordinates")

        # # Predicted robot
        # plt.scatter(
        #     pred_world[0],
        #     pred_world[1],
        #     color="red",
        #     s=80,
        #     label="Prediction",
        # )

        # # Ground truth robot
        # plt.scatter(
        #     gt_world[0],
        #     gt_world[1],
        #     color="green",
        #     s=80,
        #     label="Ground Truth",
        # )

        # plt.xlabel("X (m)")
        # plt.ylabel("Y (m)")
        # plt.title("Robot Position in World Frame")
        # plt.grid(True)
        # plt.axis("equal")
        # plt.legend()

        # plt.xlim(-5, 5)
        # plt.ylim(-5, 5)

        plt.show()