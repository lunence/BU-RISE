import torch
from . import config

def train_one_epoch(model, optimizer, data_loader, device, class_criterion, center_criterion, orientation_criterion):
    # training loop
    model.train()
    total_train_loss = 0.0
    total_center_loss = 0.0
    total_ce_loss = 0.0
    total_orientation_loss = 0.0
    total_class_loss = 0.0
    total = 0
    pose_correct = 0
    orientation_correct = 0
    combined_correct = 0


    for images, targets in data_loader:
        images = images.to(device)
        targets = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in targets.items()
        }

        optimizer.zero_grad()

        logits = model(images)

        robot_mask = targets["class"] == 1

        class_loss = class_criterion(logits["class"], targets["class"])

        center_loss = torch.tensor(0.0, device=device)
        orientation_loss = torch.tensor(0.0, device=device)
        ce_loss = torch.tensor(0.0, device=device)

        if robot_mask.any():
            center_loss = config.CENTER_LOSS_WEIGHT * center_criterion(
                logits["center"][robot_mask],
                targets["center"][robot_mask]
            )

            orientation_loss = config.ORIENTATION_LOSS_WEIGHT * orientation_criterion(
                logits["orientation"][robot_mask],
                targets["orientation"][robot_mask]
            )

            ce_loss = config.CE_LOSS_WEIGHT * class_criterion(
                logits["orientation"][robot_mask],
                targets["orientation"][robot_mask]
            )

        loss = class_loss + center_loss + orientation_loss + ce_loss

        loss.backward()
        optimizer.step()

        if robot_mask.any():
            pred_center = logits["center"][robot_mask]
            gt_center = targets["center"][robot_mask]

            pred_orientation = logits["orientation"][robot_mask].argmax(dim=1)
            gt_orientation = targets["orientation"][robot_mask]

            scale = torch.tensor([config.WIDTH, config.HEIGHT], device=device)

            center_error = torch.norm(
                (pred_center - gt_center) * scale,
                dim=1
            )

            # change range for center correct here
            center_is_correct = center_error <= config.CENTER_CORRECT_RANGE
            orientation_is_correct = pred_orientation == gt_orientation
            combined_is_correct = center_is_correct & orientation_is_correct

            total += robot_mask.sum().item()
            pose_correct += int(center_is_correct.sum().item())
            orientation_correct += int(orientation_is_correct.sum().item())
            combined_correct += int(combined_is_correct.sum().item())

        total_train_loss += loss.item()
        total_center_loss += center_loss.item()
        total_orientation_loss += orientation_loss.item()
        total_ce_loss += ce_loss.item()
        total_class_loss += class_loss.item()

    train_loss = total_train_loss / len(data_loader)
    train_accuracy = combined_correct / total if total > 0 else 0.0

    return (
        train_loss, 
        train_accuracy,     
        total_ce_loss / len(data_loader),
        total_center_loss / len(data_loader),
        total_orientation_loss / len(data_loader),
        total_class_loss / len(data_loader),
    )