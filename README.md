# Shapley
Privacy-Preserving Contribution Evaluation

The code is in the folder code_for_ppce

The file to run is the one call games.py

#results from running the file games.py
At the end of this file you can change the number of clients, alpha (NIID or more IID), global iterations, and fed iteratitions. 
the function simulation_global_iter() will save the plots for each one of the case in folders like f'./PLOTS/{self.dataset_name}/{self.num_clients}clients({self.alpha})'. 
(that you might to create them). From those plots we get the means and stds for each metric. It also will same the result in the files .log through loggers

#About the data 
The brain data set can be obtain from : https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset 
once download the dataset, you need to rewrite the path in the file data_partition.py in the entry source_dir at the begging of the file (the paths for mnist, cifar are also there, but need only brain data). To this file to be run we need, torch, torchvision. 

The file for  breast cancer data set (data_breast.py) is much simpler, you might not even will run it.

#About the models
I think I do not need specify things about the file modelos.py

#About Clients and federation 
In the file client_fed.py are the classes for the clients and the federation. As long we stay with model_type="CNN_brain", and data_name=BRAIN, I think it is okey.

#package.
you will need: torch, matplotlib, torchvision, numpy, tqdm , logging, sklearn, scipy, seaborn

#device
at the begiinig of the files games.py and client_fed.py you need to set the device to use (i.e. mps, cuda, ....) 

#utils
the file utils.py contains the functions to compute the sv, ppce, make the plots, compute the correlation coeficients

####To be run on the games.py file (change this configuration at the of the file)

The cases or resutls for the brain cancer data set we are missing are 

(num_cli,alpha,iter_global,iter_fed)\in {(6,0.5,10,15),(6,0.5,10,5),(9,0.5,10,10)}

We mention on the experiment setup that we add gaussian noise to the IID case, but I think we can just increase the parameter alpha in the direchtle partition 
and run the cases in the tables of the overleaf file for the IID case. 






