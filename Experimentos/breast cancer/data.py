import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
from torch.utils.data import random_split
from torch.utils.data import ConcatDataset


def load_data():#write name of the data as an input
    scaler = StandardScaler()
    # Load breast cancer dataset
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Divide the dataset into three equal parts
    split1, split2, split3 = np.array_split(df, 3)

    # Extract features and labels
    X_split1, y_split1 = split1.iloc[:, :-1].values, split1.iloc[:, -1].values
    X_split2, y_split2 = split2.iloc[:, :-1].values, split2.iloc[:, -1].values
    X_split3, y_split3 = split3.iloc[:, :-1].values, split3.iloc[:, -1].values

    X_standard=[scaler.fit_transform(data) for data in [X_split1,X_split2,X_split3]]
    y_targets=[y_split1,y_split2,y_split3]

    return X_standard, y_targets

def loader_client(X,y,noise_levels=[0.1,0.3,0.5]):
    clients={}
    for client in range(len(noise_levels)):
        noise_in_data = np.random.normal(0,noise_levels[client],size=X[client].shape)
        X_noisy = X[client] + noise_in_data
        X_train, X_test, y_train, y_test = train_test_split(X_noisy, y[client], test_size=0.2, random_state=42)
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_test_tensor=torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
        data_train=TensorDataset(X_train_tensor, y_train_tensor)
        data_test=TensorDataset(X_test_tensor, y_test_tensor)
        clients[client]=[DataLoader(data_train, batch_size=32, shuffle=True), DataLoader(data_test, batch_size=32, shuffle=False)]
    return clients

# Function to split dataset into train and test
def split_dataset(dataset, train_ratio=0.8):
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    return random_split(dataset, [train_size, test_size])


def NOIIDD():
    data = load_breast_cancer()
    X, y = data.data, data.target  # X: Features, y: Labels (0 = Malignant, 1 = Benign)
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    idx_malignant = np.where(y == 0)[0]  # Malignant indices
    idx_benign = np.where(y == 1)[0]
    np.random.shuffle(idx_malignant)
    np.random.shuffle(idx_benign)
    # Split indices for each client
    malignant_split = np.split(idx_malignant, [int(0.7 * len(idx_malignant)), int(0.85 * len(idx_malignant))])
    benign_split = np.split(idx_benign, [int(0.7 * len(idx_benign)), int(0.85 * len(idx_benign))])

    # Assign data to clients
    client1_idx = np.concatenate([malignant_split[0], benign_split[2]])  # Mostly Malignant
    client2_idx = np.concatenate([benign_split[0], malignant_split[2]])  # Mostly Benign
    client3_idx = np.concatenate([malignant_split[1], benign_split[1]])  # Balanced mix

    data_c1=TensorDataset(X_tensor[client1_idx],y_tensor[client1_idx]) 
    data_c2=TensorDataset(X_tensor[client2_idx],y_tensor[client2_idx])
    data_c3=TensorDataset(X_tensor[client3_idx],y_tensor[client3_idx])

    # Split each client dataset
    client1_train, client1_test = split_dataset(data_c1)
    client2_train, client2_test = split_dataset(data_c2)
    client3_train, client3_test = split_dataset(data_c3)

    clients ={
        0:[DataLoader(client1_train, batch_size=32, shuffle=True), DataLoader(client1_test, batch_size=32, shuffle=False)],
        1:[DataLoader(client2_train, batch_size=32, shuffle=True), DataLoader(client2_test, batch_size=32, shuffle=False)],
        2:[DataLoader(client3_train, batch_size=32, shuffle=True), DataLoader(client3_test, batch_size=32, shuffle=False)]
    }
    return clients



def dirichlet_partition(num_clients = 3,alpha = 0.5 ):
    # Load Wisconsin Breast Cancer dataset
    data = load_breast_cancer()
    X, y = data.data, data.target  # Features and labels

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    num_samples = len(X_tensor)

    
    dirichlet_dist = np.random.dirichlet([alpha] * num_clients, num_samples)
    client_indices = [[] for _ in range(num_clients)]

    # Assign samples to clients based on Dirichlet proportions
    for i, row in enumerate(dirichlet_dist):
        client_idx = np.argmax(row)  # Assign to client with highest probability
        client_indices[client_idx].append(i)

    data_clients=[TensorDataset(X_tensor[client_indices[i]],y_tensor[client_indices[i]]) for i in range(num_clients)]


    client1_train, client1_test = split_dataset(data_clients[0])
    client2_train, client2_test = split_dataset(data_clients[1])
    client3_train, client3_test = split_dataset(data_clients[2])

    clients ={
        0:[DataLoader(client1_train, batch_size=32, shuffle=True), DataLoader(client1_test, batch_size=32, shuffle=False)],
        1:[DataLoader(client2_train, batch_size=32, shuffle=True), DataLoader(client2_test, batch_size=32, shuffle=False)],
        2:[DataLoader(client3_train, batch_size=32, shuffle=True), DataLoader(client3_test, batch_size=32, shuffle=False)]
    }
    
    return clients


def commun_test_set(dict):
        common=[]
        for client in dict.keys():
            local_loader=dict[client][1]
            common.append(local_loader.dataset)
        combined=ConcatDataset(common)
        combined_dataloader = DataLoader(combined, batch_size=32, shuffle=True)
        return combined_dataloader




    

if __name__ == "__main__":
    #data=load_data()
    #clientes=loader_client(data[0],data[1],[0, 0, 0])
    clientes=dirichlet_partition(3,0.5)
    for i in clientes.keys():
        print(f"Client {i}")
        print(f"Train size: {len(clientes[i][0].dataset)}")
        print(f"Test size: {len(clientes[i][1].dataset)}")
        print("\n")

    # train_loader = Clients[0][0]
    # for X_batch, y_batch in train_loader: 
    #     print(X_batch.shape)
    #     print(y_batch.shape)
        # break
 #   X, y=  load_data()
 #  client_data= loader_client(X, y)




 




