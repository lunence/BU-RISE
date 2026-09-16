import torch
from . import config
def calculate_val_accuracy(model, device, data_loader_test, class_criterion, center_criterion, orientation_criterion):
    # calculate avg validation loss and avg loss for each detection head (class, center, orientation)
    total_val_loss = 0.0
    total_ce_loss = 0.0
    total_center_loss = 0.0
    total_orientation_loss = 0.0
    total_class_loss = 0.0
    
    with torch.no_grad():
        for images, targets in data_loader_test:
            logits = model(images.to(device))

            robot_mask = targets["class"] == 1
            
            total_class_loss = class_criterion(logits["class"], targets["class"])
    
            total_center_loss = torch.tensor(0.0, device=device)
            total_orientation_loss = torch.tensor(0.0, device=device)
            total_ce_loss = torch.tensor(0.0, device=device)
    
            if robot_mask.any():                
                total_ce_loss += config.CE_LOSS_WEIGHT * class_criterion(logits["orientation"][robot_mask], targets["orientation"][robot_mask])
                total_center_loss += config.CENTER_LOSS_WEIGHT * center_criterion(logits["center"][robot_mask], targets["center"][robot_mask])
                total_orientation_loss += config.ORIENTATION_LOSS_WEIGHT * orientation_criterion(logits["orientation"][robot_mask], targets["orientation"][robot_mask])

            total_val_loss += total_center_loss + total_orientation_loss + total_ce_loss + total_class_loss
    val_loss = total_val_loss / len(data_loader_test)
    ce_loss = total_ce_loss / len(data_loader_test)
    center_loss = total_center_loss / len(data_loader_test)
    orientation_loss = total_orientation_loss / len(data_loader_test)
    class_loss = total_class_loss / len(data_loader_test)
    return val_loss, ce_loss, center_loss, orientation_loss, class_loss