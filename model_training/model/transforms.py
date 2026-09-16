import torch
from torchvision.transforms import v2 as T

def get_transforms():
    transforms = []
    
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    transforms.append(T.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    ))

    return T.Compose(transforms)