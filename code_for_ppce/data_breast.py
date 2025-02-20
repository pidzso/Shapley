import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



np.random.seed(42)
torch.manual_seed(42)
# Function to split dataset into train and test
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def dirichlet_partition(num_clients=5, alpha=0.5, test_size=0.2):
    # Load the dataset
    data = load_breast_cancer()
    X, y = data.data, data.target

    # 2Normalize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 3Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    num_samples = len(X_train_tensor)

    
    dirichlet_dist = np.random.dirichlet([alpha] * num_clients, num_samples)
    client_indices = [[] for _ in range(num_clients)]

    # Assign samples to clients based on Dirichlet proportions
    for i, row in enumerate(dirichlet_dist):
        client_idx = np.argmax(row)  # Assign to client with highest probability
        client_indices[client_idx].append(i)

    data_clients=[TensorDataset(X_train_tensor[client_indices[i]],y_train_tensor[client_indices[i]]) for i in range(num_clients)]
    test_set=TensorDataset(X_test_tensor,y_test_tensor)
    clients=[DataLoader(data_clients[i], batch_size=32, shuffle=True) for i in range(num_clients)]
    test_dataloader= DataLoader(test_set, batch_size=32, shuffle=True)

    return clients,test_dataloader 



if __name__ == "__main__":
    for i in range(20):
        a=1+i
    print(a)
