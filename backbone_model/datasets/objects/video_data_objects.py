import torch
from backbone_model.simple_model_modified.dataset import Dataset
from backbone_model.util.config import TEST_SIZE
import json
from backbone_model.simple_model_modified.transforms import get_transforms

device = torch.device('cpu')

#set up ground truth data for training and testing
ground_truth = []
with open("backbone_model/datasets/video_dataset1/ground_truth.json", "r") as file:
    ground_truth = json.load(file)

# GridNet uses fixed-size images and scalar class targets, so the default
# DataLoader collation produces image batches [B, C, H, W] and targets [B].
dataset = Dataset('backbone_model/datasets/video_dataset1/frames', ground_truth, get_transforms())
dataset_test = Dataset('backbone_model/datasets/video_dataset1/frames', ground_truth, get_transforms())

#make list of same size as dataset and randomize order
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-TEST_SIZE]) 
#assign subset of last 50 of list for test
dataset_test = torch.utils.data.Subset(dataset_test, indices[-TEST_SIZE:])

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

