import torch 
from model_training.model.dataset import Dataset 
from model_training.util.config import TEST_SIZE 
import json 
from model_training.model.transforms import get_transforms 

def get_data_loaders(dataset_path, batch_size=32): 
    """ Create training and testing DataLoaders for a dataset. 
    
    Args: 
        dataset_path: Path to the dataset folder containing frames/ and ground_truth.json 
        
        batch_size: Batch size for the DataLoader 
    
    Returns: 
        data_loader: Training DataLoader 
        data_loader_test: Testing DataLoader """ 

    # Load ground truth 
    with open(f"{dataset_path}/ground_truth.json", "r") as file: 
        ground_truth = json.load(file) 

    # Create datasets 
    dataset = Dataset( 
        f"{dataset_path}/frames", 
        ground_truth, 
        get_transforms() 
    ) 

    dataset_test = Dataset( 
        f"{dataset_path}/frames", 
        ground_truth, 
        get_transforms() 
    ) 

    # Create random train/test split 
    indices = torch.randperm(len(dataset)).tolist() 

    train_indices = indices[:-TEST_SIZE] 
    test_indices = indices[-TEST_SIZE:] 

    dataset = torch.utils.data.Subset(dataset, train_indices) 
    dataset_test = torch.utils.data.Subset(dataset_test, test_indices) 

    # Create DataLoaders 
    data_loader = torch.utils.data.DataLoader( 
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
    ) 
    
    data_loader_test = torch.utils.data.DataLoader( 
        dataset_test, 
        batch_size=batch_size, 
        shuffle=False, 
    ) 

    return data_loader, data_loader_test