import torch
from torchvision.transforms import v2 as T

def get_transforms():
    return T.Compose([
        T.ToImage(),                      # converts ndarray -> TVTensor
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])