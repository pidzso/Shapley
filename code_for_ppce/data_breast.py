import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import  ConcatDataset,DataLoader, TensorDataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



np.random.seed(42)
torch.manual_seed(42)
# Function to split dataset into train and test


def add_gaussian_noise(data, mean=0.0, std=0.1):
    """
    Adds Gaussian noise to the input data.
    
    Args:
        data (torch.Tensor): Input data.
        mean (float): Mean of Gaussian noise.
        std (float): Standard deviation of Gaussian noise.
    
    Returns:
        torch.Tensor: Noisy data.
    """
    noise = torch.normal(mean=torch.full_like(data, mean), std=torch.full_like(data, std))
    return data + noise


def partition_data(num_clients=5, alpha=0.5, local_test_size=0.2, noise_levels=None, partition_type='N-IID'):
    """
    Partitions data among clients using IID or Non-IID partitioning, adds Gaussian noise to local training data,
    and creates a local test set for each client.

    Args:
        num_clients (int): Number of clients.
        alpha (float): Dirichlet parameter (smaller -> more non-iid).
        test_size (float): Fraction of total test data.
        local_test_size (float): Fraction of each client's local test set.
        noise_levels (list): List of noise standard deviations for each client.
        partition_type (str): Type of partitioning ('IID' or 'N-IID').

    Returns:
        train_dataloaders (list): List of DataLoaders for each client's training set.
        test_dataloaders (list): List of DataLoaders for each client's local test set.
    """
    # Load dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Normalize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split global train/test
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X, dtype=torch.float32)
    y_train_tensor = torch.tensor(y, dtype=torch.float32)
    
    num_samples = len(X_train_tensor)
    
    # Define noise levels if not provided
    if noise_levels is None:
        noise_levels = [40*(i/4) for i in range(num_clients)]#[1,10,20]#[1,20,30]#np.linspace(0.1, 10, num_clients)  # Different noise levels for each client
    
    client_train_dataloaders = []
    client_test_dataloaders = []
    
    if partition_type == 'N-IID':
        # Dirichlet distribution for non-IID partitioning
        dirichlet_dist = np.random.dirichlet([alpha] * num_clients, num_samples)
        client_indices = [[] for _ in range(num_clients)]
        
        for i, row in enumerate(dirichlet_dist):
            client_idx = np.argmax(row)  # Assign sample to client with highest probability
            client_indices[client_idx].append(i)
    
    elif partition_type == 'IID':
        indices = np.random.permutation(num_samples)
        client_indices = np.array_split(indices, num_clients)  # Evenly distribute indices
    
    else:
        raise ValueError("Invalid partition_type. Choose either 'IID' or 'N-IID'.")
    
    # Split each client's data into local train/test
    for i in range(num_clients):
        client_X = X_train_tensor[client_indices[i]]
        client_y = y_train_tensor[client_indices[i]]
        
        X_train_local, X_test_local, y_train_local, y_test_local = train_test_split(
            client_X, client_y, test_size=local_test_size, stratify=client_y, random_state=42)
        
        # Add Gaussian noise only to the local training set
        X_train_local_noisy = add_gaussian_noise(X_train_local, std=noise_levels[i])
        
        train_dataset = TensorDataset(X_train_local_noisy, y_train_local)
        test_dataset = TensorDataset(X_test_local, y_test_local)
        
        client_train_dataloaders.append(DataLoader(train_dataset, batch_size=32, shuffle=True))
        client_test_dataloaders.append(DataLoader(test_dataset, batch_size=32, shuffle=False))
    
    client_data={}
    for client in range(num_clients):
        client_data[client] = {"train_loader": client_train_dataloaders[client], "test_loader": client_test_dataloaders[client]}
    return client_data


def commun_test_set(dict):
        common=[]
        for client in dict.keys():
            local_loader=dict[client]["test_loader"]
            common.append(local_loader.dataset)
        combined=ConcatDataset(common)
        combined_dataloader = DataLoader(combined, batch_size=32, shuffle=True)
        return combined_dataloader

#if __name__ == "__main__":
    #def_=partition_data(num_clients=3, alpha=0.5, local_test_size=0.2, noise_levels=None, partition_type='N-IID')
    #common=commun_test_set(def_)
    #print(len(common.dataset))
