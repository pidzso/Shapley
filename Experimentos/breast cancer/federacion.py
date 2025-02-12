import matplotlib.pyplot as plt
import random
import torch
import numpy as np
import torch.nn as nn
import math
from tqdm import tqdm 
from model import LogisticRegressionModel
from data import load_data, loader_client, NOIIDD, dirichlet_partition
from data import commun_test_set
#from guassian_data import noisy_clients, data_loaders

random.seed(5)
np.random.seed(5)
torch.manual_seed(5)


torch.cuda.is_available()
# dev = "cpu"
dev = "mps" if torch.backends.mps.is_available() else "cpu"
dev = torch.device(dev)

class Clientclasi(LogisticRegressionModel):
    def __init__(self, input_dim, client_data) -> None:
        super().__init__(input_dim)
        self.local_training_loader = client_data[0]
        self.local_testing = client_data[1]
        self.client_size = len(self.local_training_loader.dataset) + len(self.local_testing.dataset)

    def evaluacion_local(self, test_set):
        return self.evaluation(test_set)

class Federation(LogisticRegressionModel):
    def __init__(self, input_dim, dict_clients_data) -> None:
        super().__init__(input_dim)
        self.input_dim = input_dim
        self.num_clientes = len(dict_clients_data)
        self.datos_client = dict_clients_data
        self.lista_clientes = [Clientclasi(input_dim, self.datos_client[i]) for i in self.datos_client.keys()]

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
    input_dim = 30
    clientes =  dirichlet_partition()
    #data=load_data()
    #clientes=loader_client(data[0],data[1],[0.1, 0.3, 0.9])
    common_dataset = commun_test_set(clientes)
    federation = Federation(input_dim, clientes)
    two=federation.federated_averaging(3, [0,1,2])
    print(federation.evaluacion(common_dataset))
   
