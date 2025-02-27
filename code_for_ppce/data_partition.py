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
from torch.utils.data import random_split
from data_breast import commun_test_set

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


def partition_data(dataname, num_clients, alpha=0.5, seed=42, train_ratio=0.8, partition_type='N-IID'):
    np.random.seed(seed)
    
    if dataname == "MNIST":
        dataset = datasets.MNIST(root=datamnist, train=True, download=True, transform=transforms.ToTensor())
    elif dataname == "CIFAR10":
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    elif dataname == "BRAIN":
        dataset = torchvision.datasets.ImageFolder(source_dir, transform=transform_brain)
    else:
        raise ValueError("Unsupported dataset")
    
    labels = np.array(dataset.targets)
    num_samples = len(labels)
    num_classes = len(np.unique(labels))
    client_indices = defaultdict(list)
    
    if partition_type == 'N-IID':
        indices_per_class = {c: np.where(labels == c)[0] for c in range(num_classes)}
        for c in range(num_classes):
            np.random.shuffle(indices_per_class[c])
            num_samples_class = len(indices_per_class[c])
            proportions = np.random.dirichlet(alpha * np.ones(num_clients))
            proportions = (np.array(proportions) * num_samples_class).astype(int)
            while proportions.sum() < num_samples_class:
                proportions[np.argmax(proportions)] += 1
            while proportions.sum() > num_samples_class:
                proportions[np.argmax(proportions)] -= 1
            start = 0
            for i in range(num_clients):
                end = start + proportions[i]
                client_indices[i].extend(indices_per_class[c][start:end])
                start = end
    else:  # IID case
        shuffled_indices = np.random.permutation(num_samples)
        split_sizes = [num_samples // num_clients] * num_clients
        for i in range(num_samples % num_clients):
            split_sizes[i] += 1
        client_splits = np.split(shuffled_indices, np.cumsum(split_sizes)[:-1])
        for i, split in enumerate(client_splits):
            client_indices[i] = split.tolist()
    
    client_train_test = {}
    for client, indices in client_indices.items():
        train_size = int(len(indices) * train_ratio)
        train_indices, test_indices = indices[:train_size], indices[train_size:]
        client_train_test[client] = {"train": train_indices, "test": test_indices}
    
    return client_train_test, dataset

def data_for_clients(data_name, num_clients, alpha=0.5, train_ratio=0.8, partition_type='N-IID'):
    client_partitions, dataset = partition_data(data_name, num_clients, alpha, train_ratio=train_ratio, partition_type=partition_type)
    client_data = {}
    
    for client, indices in client_partitions.items():
        train_loader = DataLoader(Subset(dataset, indices['train']), batch_size=32, shuffle=True)
        test_loader = DataLoader(Subset(dataset, indices['test']), batch_size=32, shuffle=False)
        client_data[client] = {"train_loader": train_loader, "test_loader": test_loader}
    
    return client_data
