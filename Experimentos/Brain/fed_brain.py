import matplotlib.pyplot as plt
import random
import torch
import numpy as np
import torch.nn as nn
import math
from tqdm import tqdm 
from model_cnn import SimpleCNN
from data_brain import data_for_clients, commun_test_set

random.seed(5)
np.random.seed(5)
torch.manual_seed(5)


torch.cuda.is_available()
# dev = "cpu"
dev = "mps" if torch.backends.mps.is_available() else "cpu"
dev = torch.device(dev)

class Clientclasi(SimpleCNN):
    def __init__(self, input_channels, num_classes, client_data) -> None:
        super().__init__(input_channels, num_classes)
        self.local_training_loader = client_data[0]
        self.local_testing = client_data[1]
        self.client_size = len(self.local_training_loader.dataset) + len(self.local_testing.dataset)

    def evaluacion_local(self, test_set):
        return self.evaluation(test_set)

class Federation(SimpleCNN):
    def __init__(self, input_channels, num_classes, dict_clients_data) -> None:
        super().__init__(input_channels, num_classes)
    
        self.num_clientes = len(dict_clients_data)
        self.datos_client = dict_clients_data
        self.lista_clientes = [Clientclasi(input_channels, num_classes, self.datos_client[i]) for i in self.datos_client.keys()]

    def total_size(self):
        size = 0
        for client in self.lista_clientes:
            size += client.client_size
        return size

    def clientes(self):
        length = len(self.lista_clientes)
        return list(range(length))

    def aggregate(self, List_clients):
        pesos_cliente = [client.parameters() for client in List_clients]
        data = []
        for params_clients in zip(*pesos_cliente):
            data.append(sum(param.data for param in params_clients) / len(List_clients))
        return data
    
    def weighted_aggre(self,List_clients):
        pesos_cliente = [client.parameters() for client in List_clients] 
        total_data = sum(cliente.client_size for cliente in List_clients)
        data_amount=[participant.client_size for participant in List_clients]
        data = []
        for params_clients in zip(*pesos_cliente):
            order_para=[]
            for param, amout in zip(params_clients,tuple(data_amount)):
                order_para.append((amout/total_data)*param.data)
            data.append(sum(order_para))
        return data
    
    def federated_averaging(self, num_iterations, conjunto_clientes,num_epochs=20):
        conj_clientes = [self.lista_clientes[i] for i in conjunto_clientes]
        iteration = 0
        while iteration <= num_iterations - 1:
            for client in conj_clientes:
                weights_modelo = self.state_dict()
                client.load_state_dict(weights_modelo)
                client.fit(num_epochs, client.local_training_loader)
            tqdm.write(f"fed iteration {iteration}")
            params = self.weighted_aggre(conj_clientes)
            self.set_parameters(params)
            iteration += 1
        #return self.evaluacion(test_set)  
        

    def evaluacion(self, testing):
        return self.evaluation(testing)

if __name__ == "__main__":
    input_channels=3
    num_classes=4
    dic_labels={0:[100,100,0,0],1:[0,100,100,0],2:[0,0,0,100]}
    dic_loaders=data_for_clients(dic_labels)
    test_loader_co=commun_test_set(dic_loaders) 
    federation=Federation(input_channels, num_classes,dic_loaders)
    federation.federated_averaging(3,[0,1,2], num_epochs=5, plot=False)
    print(federation.evaluacion(test_loader_co))


   
   
