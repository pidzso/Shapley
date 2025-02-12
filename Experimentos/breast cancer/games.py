import matplotlib.pyplot as plt
import random
import torch
import numpy as np
import torch.nn as nn
import math
from tqdm import tqdm 
from itertools import combinations
from data import load_data, loader_client, NOIIDD, dirichlet_partition
from data import commun_test_set
import logging
from federacion import Federation


torch.cuda.is_available()
# dev = "cpu"
dev = "mps" if torch.backends.mps.is_available() else "cpu"
dev = torch.device(dev)

logging.basicConfig(level=logging.INFO, filename="results_breast.log",format="%(asctime)s-%(levelname)s-%(message)s")
logger = logging.getLogger()



class games(Federation):
    def __init__(self, input_dim, dict_clients_dataa):
        super().__init__(input_dim, dict_clients_dataa)

    
        self.cl=self.clientes()
        self.tot_size=self.total_size()

        self.clients_training=[dict_clients_dataa[x][0] for x in  self.cl]
        self.clients_testing=[dict_clients_dataa[x][1] for x in  self.cl]
        self.common_dataset=commun_test_set(dict_clients_dataa)


    def retrainin_game(self,iterations, con_clientes, testing_loader,epochs=5):
        if len(con_clientes) == 0:
            return 0
        self.reset_parameters()
        self.federated_averaging(iterations,con_clientes,num_epochs=epochs)
        worth = self.evaluacion(testing_loader)
        return worth
    
    
    def coalitional_values(self,iteration,testing_loader, epochs=5):
        d = dict()
        d[()]=0
        for client in self.cl:
            self.lista_clientes[client].fit(epochs, self.clients_training[client])
            d[tuple([client])] = self.lista_clientes[client].evaluacion_local(testing_loader)
            self.lista_clientes[client].reset_parameters()
        for subset_len in range(2,len(self.cl)+1):
            for subset_idx in combinations(self.cl, subset_len):
                d[subset_idx] = self.retrainin_game(iteration,list(subset_idx), testing_loader,epochs)
            tqdm.write(f"subset size {subset_len}")
        logger.info(f"True coalitional values: {d}")
        return d
    
    def coalitional_values_II(self,iteration,testing_loader, epochs=5):
        self.reset_parameters()
        d = dict()
        d[()]=0
        for i in range(iteration):
            self.traing_together(self.lista_clientes, epochs)
        for client in self.cl:
            d[tuple([client])] = self.lista_clientes[client].evaluacion_local(testing_loader)
        for subset_len in range(2,len(self.cl)+1):
            for subset_idx in combinations(self.cl, subset_len):
                part=[self.lista_clientes[i] for i in subset_idx]
                params = self.aggregate(part)
                self.set_parameters(params)
                d[subset_idx] = self.evaluacion(testing_loader)
            tqdm.write(f"subset size {subset_len}")
        logger.info(f"at coalitional values: {d}")
        return d
    
    
    
    def traing_together(self,conj_clientes, num_epochs):
        for client in conj_clientes:
            weights_modelo = self.state_dict()
            client.load_state_dict(weights_modelo)
            client.fit(num_epochs, client.local_training_loader)
        params = self.weighted_aggre(conj_clientes)
        self.set_parameters(params)


    def combine_plot(self,groups=list,categories=list,name=str):
        # Data setup
        #categories = ['SV', 'L1O', 'I1I', 'EEE']
        matrix=np.array(groups)


        # Number of categories and width for each bar
        x = np.arange(len(categories))
        width = 0.2  # Adjusts spacing between bars

        # Plotting each group
        plt.bar(x - 1.5 * width, matrix[:,0], width, label='hospital A')
        plt.bar(x - 0.5 * width, matrix[:,1], width, label='hospital B')
        plt.bar(x + 0.5 * width, matrix[:,2], width, label='hospital C')


       # Labels, legend, and tick marks
        plt.xlabel('Metric')
        plt.ylabel('Scores')
        plt.title(f"{name}")
        plt.xticks(x, categories)
        plt.legend()
        plt.grid()
        plt.savefig(f'/Users/delio/Documents/Working projects/Balazs/Experiments/breast_cancer/figures/{name}')

    


    def transform_dict_to_lists(self,dic,clients):
        #n = 3  # Number of elements in the binary representation
        sets = []
        accuracies = []

        for key, value in dic.items():
            binary_list = [0] * clients
            for index in key:
                binary_list[index] = 1
            sets.append(binary_list)
            accuracies.append(value)

        return sets, accuracies


    def shapley(self, clients, groups, acc,case=str):
        '''
        compute the Shapley Value
            (input)  clients: client number
            (input)  groups:  coalitions with binary assignment matrix, e.g. for 2 players [[0,0],[1,0],[0,1],[1,1]]
            (input)  acc:     accuracies of the corresponding groups, e.g., for two players [a, b, c, d]
            (output) score:   Shapley value of the players
        '''
        scores = np.zeros(clients)
        for i in range(clients):
            tmp = 0
            for j, subset in enumerate(groups):
                if subset[i] == 0:
                    continue
                subset_without_i = np.copy(subset)
                subset_without_i[i] = 0
                subset_index = j
                idx = [k for k, l in enumerate(groups) if all(x == y for x, y in zip(l, subset_without_i))]
                subset_without_i_index = idx[0]
                marginal_contribution = acc[subset_index] - acc[subset_without_i_index]
                subset_size = np.sum(subset) - 1
                weight = (math.factorial(subset_size) * math.factorial(clients - subset_size - 1)) / math.factorial(clients)
                tmp += weight * marginal_contribution
            scores[i] = tmp
        logger.info(f"{case}: {scores}")
        return scores



    def ppce(self, clients, groups, acc):
        '''
        compute the privacy-preserving contribution scores
            (input)  clients: client number
            (input)  groups:  coalitions with binary assignment matrix, e.g. for 2 players [[0,0],[1,0],[0,1],[1,1]]
            (input)  acc:     accuracies of the corresponding groups, e.g., for two players [a, b, c, d]
            (output) i1i:     include-one-in
            (output) l1o:     leave-one-out
            (output) ieei:    include-everybody-else-in
            (output) leeo:    leave-everybody-else-out
        '''
        i1i = np.zeros(clients)
        l1o = np.zeros(clients)
        ieei = np.zeros(clients)
        leeo = np.zeros(clients)
        include = np.zeros(clients)
        leave = np.zeros(clients)
        for i, subset in enumerate(groups):
            if np.sum(subset) == 0:
                null = acc[i]
            if np.sum(subset) == clients:
                grand = acc[i]
            if np.sum(subset) == 1:
                tmp = [j for j, k in enumerate(subset) if k == 1]
                include[tmp[0]] = acc[i]
            if np.sum(subset) == clients - 1:
                tmp = [j for j, k in enumerate(subset) if k == 0]
                leave[tmp[0]] = acc[i]
        for i in range(clients):
            i1i[i] = include[i] - null
            l1o[i] = grand - leave[i]
            for j in range(clients):
                ieei[i] += leave[j] - null
                leeo[i] += grand - include[j]
            ieei[i] -= leave[i] - null
            leeo[i] -= grand - include[i]
        ie2i= ieei / (clients - 1) ** 2
        le2o= leeo / (clients - 1) ** 2
        logger.info(f"i1i: {i1i.tolist()}")
        logger.info(f"l10: {l1o.tolist()}")
        logger.info(f"ie2i: {ie2i.tolist()}")
        logger.info(f"le2o: {le2o.tolist()}")
        return [i1i.tolist(), l1o.tolist(), ie2i.tolist(), le2o.tolist()]

if __name__ == "__main__":
    #input of the model 
    input_dim=30
    #####case to plot 
    case="dirichlet_II"
    ###################### data case
    clientes =  dirichlet_partition(3,0.5)
    #clientes = NOIIDD()
    #data=load_data()
    #clientes=loader_client(data[0],data[1],[1, 2, 3])
    ######common dataset
    common_dataset=commun_test_set(clientes)
    #####game instance
    juegos=games(input_dim,clientes)
    ######shapley retraining
    values_re=juegos.coalitional_values(3,common_dataset)
    transform_re=juegos.transform_dict_to_lists(values_re,3)
    shapley_re=juegos.shapley(3,transform_re[0],transform_re[1],"SV_re")
    ######shapley at round
    values_ro=juegos.coalitional_values_II(3,common_dataset)
    transform_ro=juegos.transform_dict_to_lists(values_ro,3)
    shapley_ro=juegos.shapley(3,transform_ro[0],transform_ro[1],"SV_ro")
    #######ppce
    ppce=juegos.ppce(3,transform_ro[0],transform_ro[1])
    ######plots
    games_pp=[shapley_re,shapley_ro]+ppce
    juegos.combine_plot(games_pp,["SV_re","SV_ro","I1I","L1O","IE2I","LE2O"],case)