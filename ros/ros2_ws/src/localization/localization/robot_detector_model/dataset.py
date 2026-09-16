import os
import torch
import cv2
import re
from . import config
from localization.utils import crop_to_ratio

class Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: str, ground_truth: list[dict], transform=None):
        self.root_dir = root_dir
        self.ground_truth = ground_truth
        self.transform = transform
        self.image_files = sorted(
            (f for f in os.listdir(root_dir) if f.endswith((".png", ".jpg"))),
            key=lambda filename: int(
                re.search(r"(\d+)(?=\.[^.]+$)", filename).group(1)
            ),
        )
        # print("Images:", len(self.image_files))
        # print("Ground truth:", len(self.ground_truth), len(self.ground_truth))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.image_files[idx])
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        if image.shape[:2] != (config.HEIGHT, config.WIDTH):  # (height, width)
            image = crop_to_ratio(image)
            image = cv2.resize(image, (config.WIDTH, config.HEIGHT), interpolation=cv2.INTER_AREA)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).permute(2, 0, 1)
        # image = (image - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / \
        #     torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        
        target = {}

        if idx < len(self.ground_truth) and self.ground_truth[idx]:
            # A single class represents a pair of values:
            #   class = pose_cell * ANGLE_CLASSES + orientation_bin
            # pose_cell is mapped to a 0-based index over the available pose cells,
            # and orientation_bin is a 0-based bin index in [0, ANGLE_CLASSES).
            center = self.ground_truth[idx]["center"]
            orientation_bin = self.ground_truth[idx]["orientation"] % 360 // (360 / 72)
            target["center"] = torch.tensor(center, dtype=torch.float32)
            target["orientation"] = torch.tensor(orientation_bin, dtype=torch.long)
            target["class"] = self.ground_truth[idx].get("class", 1)

        if self.transform:
            image = self.transform(image)

        return image, target
