import matplotlib.pyplot as plt
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import math
from tqdm import tqdm
from data_brain import data_for_clients, commun_test_set
import torch.nn.functional as F


torch.cuda.is_available()
# dev = "cpu"
dev = "mps" if torch.backends.mps.is_available() else "cpu"
dev = torch.device(dev)

class SimpleCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)  # Output: (32, H, W)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Output: (64, H, W)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Halves H and W
        # Fully connected layers
        self.fc1 = nn.Linear(64 * (256 // 2) * (256 // 2), 128)  # Adjust for input size
        self.fc2 = nn.Linear(128, num_classes)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001, weight_decay=0.001)

        
    def forward(self, x):
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten the tensor
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, features)
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def fit(self,num_epochs,data_loader):
        self.to(dev)
        for epoch in tqdm(range(num_epochs)):
            self.train()
            for data, target in data_loader:
                data, target = data.to(dev), target.to(dev)
                self.optimizer.zero_grad()
                output = self(data)
                loss = nn.functional.cross_entropy(output, target)
                loss.backward()
                self.optimizer.step()
              
    
    def evaluation(self,test_loader):
        self.eval()
        correct = 0
        total = 0
        i = 0
        #self.to(dev)
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(dev), target.to(dev)
                outputs = self(data)
                _, predicted = torch.max(outputs, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                i+=1
                if i > 5:# this for the batches
                    break
        return correct/total
        
    def get_model_grads(self):
        return [param.grad for param in self.parameters()]
    
    
    def reset_parameters(self):
        # self.normalizer.reset_parameters()
        # self.layers.reset_parameters()
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def set_parameters(self,params):
         for model_parametro, param in zip(self.parameters(), params):
             model_parametro.data = param
   
    
if __name__ == "__main__":
    input_channels=3
    num_classes=4
    dic_labels={0:[100,100,0,0],1:[0,100,100,0],2:[0,0,0,100]}
    dic_loaders=data_for_clients(dic_labels)
    test_loader_co=commun_test_set(dic_loaders) 
    mlp = SimpleCNN(input_channels, num_classes)
    mlp.fit(10,dic_loaders[0][0])
    eva=mlp.evaluation(test_loader_co)
    print(eva)

