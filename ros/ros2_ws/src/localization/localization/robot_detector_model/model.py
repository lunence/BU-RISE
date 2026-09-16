import torchvision
import torch
import torch.nn as nn

class GridNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = torchvision.models.mobilenet_v2(
            weights="DEFAULT"
        ).features

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.class_head = nn.Linear(256, 2)

        # Predict normalized (x, y) center
        self.center_head = nn.Linear(256, 2)

        # Predict orientation class
        self.orientation_head = nn.Linear(256, 72)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = self.shared(x)

        return {
            "class": self.class_head(x),
            "center": self.center_head(x),
            "orientation": self.orientation_head(x),
        }