import matplotlib.pyplot as plt
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import copy
import math
from tqdm import tqdm
from torchvision.transforms import ToTensor
from data_mnist import data_for_clients, commun_test_set

torch.cuda.is_available()
# dev = "cpu"
dev = "mps" if torch.backends.mps.is_available() else "cpu"
dev = torch.device(dev)

class MLP(nn.Module):
    def __init__(self, neurons):  #'self' is required
        super(MLP, self).__init__()  # <- This is required

        # Define all layers as a special list
        self.layers = nn.Sequential()
        #self.normalizer = nn.BatchNorm1d(neurons[0], affine=False, track_running_stats=False)
        #self.normalizer = nn.LayerNorm(neurons[0])
        #self.layers.append(normalizer)
        # This way, we just iteratively append all layers
        for i in range(len(neurons)-2):
            self.layers.append(nn.Linear(neurons[i], neurons[i+1]))
            self.layers.append(nn.ReLU())
        
        # Unfortunately, the last layer has a different activation function
        #(an 'if' statement could alse be introduced in the above loop)
        self.layers.append(nn.Linear(neurons[i+1], neurons[i+2]))
        self.layers.append(nn.Softmax(dim=1))
       
        # self.criterion = nn.CrossEntropyLoss()
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.001)

    def forward(self, x):
       # x = self.normalizer(x)
        x = x.view(-1, 28*28)
        x = self.layers(x)
        return x
    
    def fit(self, num_epochs, data_loader):
        # Training loop
        lenght=len(data_loader)
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
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(dev), target.to(dev)
                outputs = self(data)
                _, predicted = torch.max(outputs, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                i+=1
                if i > 5:
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
    neurons = [784,16,10]
    dic_labels={0:[30,30,0,0,0,0,0,0,0,0],1:[0,30,30,0,0,0,0,0,0,0],2:[0,0,0,30,0,0,0,0,0,0]}
    dic_loaders=data_for_clients(dic_labels)
    test_loader_co=commun_test_set(dic_loaders) 
    mlp = MLP(neurons)
    #mlp.to(dev)
    mlp.fit(10,dic_loaders[0][0])
    #mlp.plot()
    eva=mlp.evaluation(test_loader_co)
    print(eva)


