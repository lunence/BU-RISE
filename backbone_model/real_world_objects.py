import torch
from simple_model_modified.dataset import Dataset
from config import TEST_SIZE, TAG_SIZE_LIMO, APRILTAG_HEIGHT, APRILTAG_WIDTH
import json
from simple_model_modified.transforms import get_transforms


#use accelerator (offloading operations to gpu) or cpu if accelerator not available
# device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device('cpu')

device = torch.device('cpu')

#set up ground truth data for training and testing
ground_truth = []
with open("backbone_model/datasets/synthetic_dataset/ground_truth.json", "r") as file:
    ground_truth = json.load(file)

ground_truth_real_world = []
with open("backbone_model/datasets/synthetic_dataset/video/ground_truth.json", "r") as file:
    ground_truth_real_world = json.load(file)

# GridNet uses fixed-size images and scalar class targets, so the default
# DataLoader collation produces image batches [B, C, H, W] and targets [B].
dataset = Dataset('backbone_model/datasets/synthetic_dataset/images', ground_truth, get_transforms())
dataset_test = Dataset('backbone_model/datasets/synthetic_dataset/images', ground_truth, get_transforms())
dataset_real_world = Dataset('backbone_model/datasets/synthetic_dataset/video', ground_truth_real_world, get_transforms())

#make list of same size as dataset and randomize order
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-TEST_SIZE]) 
#assign subset of last 50 of list for test
dataset_test = torch.utils.data.Subset(dataset_test, indices[-TEST_SIZE:])

real_world_indices = torch.randperm(len(dataset_real_world)).tolist()
dataset_real_world = torch.utils.data.Subset(dataset_real_world, real_world_indices[:]) 

# dataset = torch.utils.data.Subset(dataset, indices[:50])
# dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

# # dataset for overfitting below
# dataset = torch.utils.data.Subset(dataset, indices[:50])
# #assign subset of last 50 of list for test
# dataset_test = torch.utils.data.Subset(dataset_test, indices[:50])


#load train and test data
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    shuffle=True, 
)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=32,
    shuffle=False,
)

