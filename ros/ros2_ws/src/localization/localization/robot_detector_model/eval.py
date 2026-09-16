import torch
from . import config

def eval(model, data_loader_test, device):
    # evaluation loop
    model.eval()

    combined_correct = 0
    total = 0
    total_images = 0
    pose_correct = 0
    orientation_correct = 0
    class_correct = 0

    all_center_error = []
    all_orientation_error = []

    scale = torch.tensor([config.WIDTH, config.HEIGHT], device=device)

    with torch.no_grad():
        for images, targets in data_loader_test:
            images = images.to(device)
            targets = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in targets.items()
            }

            logits = model(images)
            pred_class = logits["class"].argmax(dim=1)
            gt_class = targets["class"]

            class_is_correct = pred_class == gt_class
            class_correct += int(class_is_correct.sum().item())

            robot_mask = targets["class"] == 1
    
            if robot_mask.any():
                # only calculate center and orientation losses when robot detected
                pred_center = logits["center"][robot_mask]
                gt_center = targets["center"][robot_mask]

                pred_orientation = logits["orientation"][robot_mask].argmax(dim=1)
                gt_orientation = targets["orientation"][robot_mask]

                center_error = torch.norm(
                    (pred_center - gt_center) * scale,
                    dim=1
                )
                all_center_error.extend(
                    center_error.cpu().tolist()
                )

                orientation_error = torch.abs(pred_orientation - gt_orientation)
                orientation_error = torch.minimum(orientation_error, 72 - orientation_error)
                all_orientation_error.extend(
                    orientation_error.cpu().tolist()
                )

                # change range for center correct here
                center_is_correct = center_error <= config.CENTER_CORRECT_RANGE
                orientation_is_correct = pred_orientation == gt_orientation
                combined_is_correct = center_is_correct & orientation_is_correct

                total += robot_mask.sum().item()
                total_images += images.size(0)
                pose_correct += int(center_is_correct.sum().item())
                orientation_correct += int(orientation_is_correct.sum().item())
                combined_correct += int(combined_is_correct.sum().item())

                # print each prediction
                for i in range(pred_center.size(0)):
                    print(
                        f"predicted center: {pred_center[i].cpu().numpy()}, "
                        f"predicted orientation: {pred_orientation[i].item() * 5}°, "
                        f"actual center: {gt_center[i].cpu().numpy()}, "
                        f"actual orientation: {gt_orientation[i].item() * 5}°, "
                        f"predicted class: {pred_class[i].item()}, "
                        f"actual class: {gt_class[i].item()}"
                    )

    if len(all_center_error) != 0 and len(all_orientation_error) != 0:
        mean_center_error = sum(all_center_error) / len(all_center_error)
        mean_orientation_error = sum(all_orientation_error) / len(all_orientation_error)
    else:
        mean_center_error = None
        mean_orientation_error = None

    if mean_center_error is None:
        print("\nMean center error: N/A px")
    else:
        print(f"\nMean center error: {mean_center_error:.3f} px")

    if mean_orientation_error is None:
        print("Mean orientation error: N/A degrees")
    else:
        print(f"Mean orientation error: {mean_orientation_error * 5:.3f} degrees")

    accuracy = combined_correct / total if total else 0.0
    pose_accuracy = pose_correct / total if total else 0.0
    orientation_accuracy = orientation_correct / total if total else 0.0
    class_accuracy = class_correct / total_images if total_images else 0.0

    return accuracy, pose_accuracy, orientation_accuracy, class_accuracy, all_center_error, all_orientation_error