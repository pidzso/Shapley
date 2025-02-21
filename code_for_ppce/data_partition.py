import torch
import os
import numpy as np
import torch.nn as nn
import math
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Subset
import random
from collections import defaultdict
import pathlib

#path for mnist data set
datamnist= os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/mnist/'
#transform and path for brain data set
source_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/brain/'
source_dir = pathlib.Path(source_dir)
#I have cifar10 in the local folder, so no need the path here. but in case place it here

##transfor for brain data set
transform_brain = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# transforms.Compose(
#             [
#                 transforms.Resize((256,256)),
#                 transforms.RandomHorizontalFlip(p=0.5),
#                 transforms.RandomVerticalFlip(p=0.5),
#                 transforms.RandomRotation(30),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
#                 #AddGaussianNoise(mean=0.0, std=0.8)
#         ]
#         )


def partition_data_dirichlet(dataname, num_clients, alpha=0.5, seed=42):
    """
    Partitions dataset indices using a Dirichlet distribution for non-IID splits.
    
    Args:
        dataset (torchvision.datasets): Dataset object (e.g., CIFAR-10, MNIST,BRAIN).
        num_clients (int): Number of clients.
        alpha (float): Dirichlet distribution parameter (controls non-IID level).
        seed (int): Random seed for reproducibility.

    Returns:
        list: list with lists of dataset indices of each client.
    """
    np.random.seed(seed)

    if dataname=="MNIST":
        dataset=datasets.MNIST(root=datamnist, train=True, download=False, transform=ToTensor())
    elif dataname=="CIFAR10":
        transform = transforms.Compose([transforms.ToTensor()])
        dataset=datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    elif dataname=="BRAIN":
        # Define an object of the custom dataset for the train and validation.
        dataset = torchvision.datasets.ImageFolder(source_dir.joinpath("Training"), transform=transform_brain) 
        dataset.transform
    else:
        raise ValueError("Not support data set")

    # Get labels from dataset
    if hasattr(dataset, 'targets'):  # Works for CIFAR-10, MNIST
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'train_labels'):  # Some older torchvision versions
        labels = np.array(dataset.train_labels)
    else:
        raise ValueError("Dataset format not recognized, check label attribute.")

    num_classes = len(np.unique(labels))
    indices_per_class = {c: np.where(labels == c)[0] for c in range(num_classes)}

    # Dirichlet distribution to allocate data
    client_indices = defaultdict(list)
    
    for c in range(num_classes):
        np.random.shuffle(indices_per_class[c])  # Shuffle indices of class `c`
        num_samples = len(indices_per_class[c])
        
        # Sample proportions for each client from Dirichlet distribution
        proportions = np.random.dirichlet(alpha * np.ones(num_clients))
        
        # Convert proportions into actual data splits
        proportions = (np.array(proportions) * num_samples).astype(int)
        
        # Fix rounding issues to ensure sum equals num_samples
        while proportions.sum() < num_samples:
            proportions[np.argmax(proportions)] += 1
        while proportions.sum() > num_samples:
            proportions[np.argmax(proportions)] -= 1
        
        # Assign indices to each client
        start = 0
        for i in range(num_clients):
            end = start + proportions[i]
            client_indices[i].extend(indices_per_class[c][start:end])
            start = end
    
    return client_indices 

def data_loader(data_set,subset):
    loader=DataLoader(Subset(data_set,subset),batch_size=32, shuffle=True)
    return loader

def data_for_client_NoIID(data_name,num_clients,alpha=0.5):
    """
    depending on the data set output the dataloaders for each client
    """
    if data_name=="MNIST":
        dataset=datasets.MNIST(root=datamnist, train=True, download=False, transform=ToTensor())
    elif data_name=="CIFAR10":
        transform = transforms.Compose([transforms.ToTensor()])
        dataset=datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    elif data_name=="BRAIN":
        # Define an object of the custom dataset for the train and validation.
        dataset = torchvision.datasets.ImageFolder(source_dir.joinpath("Training"), transform=transform_brain) 
    else:
        raise ValueError("Not support data set")
    dict_loaders=[]
    training=partition_data_dirichlet(data_name,num_clients,alpha)
    for client in range(num_clients):
        dict_loaders.append(data_loader(dataset,training[client]))
    return dict_loaders

def commun_test_set(data_name):
    """
    just conver test into data loader depending on the chosen data set
    """
    if data_name=="MNIST":
        dataset=datasets.MNIST(root=datamnist, train=False, download=False, transform=ToTensor())
    elif data_name=="CIFAR10":
        transform = transforms.Compose([transforms.ToTensor()])
        dataset=datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    elif data_name=="BRAIN":
        # Define an object of the custom dataset for the train and validation.
        dataset = torchvision.datasets.ImageFolder(source_dir.joinpath("Testing"), transform=transform_brain)
    else:
        raise ValueError("Not support data set")
    combined_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    return combined_dataloader


if __name__ == "__main__":
    data_name="BRAIN"
    num_clients=3
    set=data_for_client_NoIID(data_name,num_clients,alpha=0.5)
    for data, labels in set[0]:
        print(f"Input shape: {data.shape}")   # Shape of input tensor
        print(f"Label shape: {labels.shape}") # Shape of label tensor
        break
    # print(len(set[1].dataset))
    # print("Classes:", dataset.classes)
    # print("Class to Index Mapping:", dataset.class_to_idx)
    # #print("First 5 Image Paths and Labels:", dataset.samples[:5])
    # print("First 5 Targets (Labels):", dataset.targets[:5])
    # # data=data_for_client_NoIID("MNIST",10)
    # # print(len(data[0].dataset))
