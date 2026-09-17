import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
import time

from model_training.model.model import GridNet
from model_training.util.config import WIDTH, HEIGHT
from model_training.util.world_frame import WorldFrame

from model_training.train_val_loop import device


def visualize(model_path, data_loader_test):

    image = cv2.imread(
        "model_training/initialization_apriltag.jpg"
    )

    worldframe = WorldFrame(image)

    model = GridNet().to(device)

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    shown = 0

    # ==================================================
    # Iterate through batches
    # ==================================================

    for images, targets in data_loader_test:

        # Process each image in the batch
        for i in range(images.size(0)):

            if shown >= 20:
                return

            image = images[i]

            target = {
                key: value[i]
                for key, value in targets.items()
            }

            # ==================================================
            # Skip samples with no valid ground truth
            # ==================================================

            if (
                torch.all(target["center"] == 0)
                and target["orientation"].item() == 0
            ):
                continue


            # ==================================================
            # Model inference
            # ==================================================

            input_image = image.unsqueeze(0).to(device)

            # Start inference timer
            start_time = time.perf_counter()

            with torch.no_grad():
                logits = model(input_image)

            # End inference timer
            end_time = time.perf_counter()

            inference_time = (
                end_time - start_time
            )

            inference_time_ms = (
                inference_time * 1000
            )

            inference_fps = (
                1 / inference_time
                if inference_time > 0
                else 0
            )


            # ==================================================
            # Get predictions
            # ==================================================

            scale = torch.tensor(
                [WIDTH, HEIGHT],
                device=device
            )

            pred_center = logits["center"][0]
            pred_center = pred_center * scale

            pred_orientation = (
                logits["orientation"][0].argmax()
            )

            gt_center = target["center"]
            gt_center = gt_center * scale

            gt_orientation = target["orientation"]


            # ==================================================
            # Convert image for plotting
            # ==================================================

            mean = torch.tensor(
                [0.485, 0.456, 0.406]
            ).view(1, 1, 3)

            std = torch.tensor(
                [0.229, 0.224, 0.225]
            ).view(1, 1, 3)

            img = image.permute(
                1, 2, 0
            ).cpu()

            img = img * std + mean

            img = img.clamp(
                0,
                1
            )

            img = (
                img.numpy() * 255
            ).astype(
                np.uint8
            ).copy()


            # ==================================================
            # Print ground truth
            # ==================================================

            print(
                "Ground Truth"
            )

            print(
                "------------"
            )

            print(
                "Centers:",
                gt_center
            )

            print(
                "Orientations (bins):",
                gt_orientation
            )

            print(
                "Orientations (angle):",
                gt_orientation * 5
            )

            print()


            # ==================================================
            # Print predictions
            # ==================================================

            print(
                "Prediction"
            )

            print(
                "----------"
            )

            print(
                "Centers:",
                pred_center
            )

            print(
                "Orientations (bins):",
                pred_orientation
            )

            print(
                "Orientations (angle):",
                pred_orientation * 5
            )


            # ==================================================
            # Print inference time
            # ==================================================

            print()

            print(
                f"Inference Time: "
                f"{inference_time_ms:.3f} ms"
            )

            print(
                f"Inference FPS: "
                f"{inference_fps:.2f}"
            )


            # ==================================================
            # Calculate errors
            # ==================================================

            orientation_error = torch.abs(
                pred_orientation - gt_orientation
            )

            orientation_error = torch.minimum(
                orientation_error * 5,
                360 - orientation_error * 5
            )

            center_error = torch.norm(
                pred_center - gt_center
            )

            print()

            print(
                "Center Error:",
                center_error
            )

            print(
                "Orientation Error:",
                orientation_error
            )


            # ==================================================
            # Convert predicted center to world coordinates
            # ==================================================

            cx, cy = (
                pred_center
                .cpu()
                .tolist()
            )

            pred_world = (
                worldframe.pixel_to_world(
                    [cx, cy]
                )
            )

            print(
                "Predicted World Coords:",
                pred_world
            )


            # ==================================================
            # Draw predicted center
            # ==================================================

            cv2.circle(
                img,
                (int(cx), int(cy)),
                radius=2,
                color=(0, 0, 255),
                thickness=-1
            )

            cv2.putText(
                img,
                f"({cx:.1f}, {cy:.1f})",
                (int(cx), int(cy - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )


            # ==================================================
            # Ground truth center
            # ==================================================

            cx, cy = (
                gt_center
                .cpu()
                .tolist()
            )

            gt_world = (
                worldframe.pixel_to_world(
                    [cx, cy]
                )
            )

            print(
                "Actual World Coords:",
                gt_world
            )

            cv2.circle(
                img,
                (int(cx), int(cy)),
                radius=2,
                color=(0, 255, 0),
                thickness=-1
            )

            cv2.putText(
                img,
                f"({cx:.1f}, {cy:.1f})",
                (int(cx), int(cy - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )


            # ==================================================
            # Display image
            # ==================================================

            plt.figure(
                "Image"
            )

            plt.imshow(
                img,
                aspect="equal"
            )

            plt.axis(
                "off"
            )

            plt.show()


            shown += 1