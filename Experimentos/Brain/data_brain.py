import pathlib
import torch
import torchvision
import torchvision.transforms as transforms # A module from the torchvision library that provides common image transformations, such as resizing, cropping, and normalization.
import pathlib
import numpy as np
from torch.utils.data import DataLoader, Subset, ConcatDataset
import random 


source_dir= '/Users/delio/Documents/Working projects/Balazs/Experiments/Brain/archive'
source_dir = pathlib.Path(source_dir)
transform = transforms.Compose(
    [
        transforms.Resize((256,256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
        #AddGaussianNoise(mean=0.0, std=0.8)
   ]
)

# Define an object of the custom dataset for the train and validation.
train_set = torchvision.datasets.ImageFolder(source_dir.joinpath("Training"), transform=transform) 
train_set.transform
test_set = torchvision.datasets.ImageFolder(source_dir.joinpath("Testing"), transform=transform)
test_set.transform



def choosing_labels(data_set,labels=list,samples=list):
    index_sambles=[]
    for etiqueta in labels:
        indices_etiqueta=[i for i, (_, label) in enumerate(data_set) if label == etiqueta]
        random_etiqueta=random.sample(indices_etiqueta,samples[labels.index(etiqueta)])
        index_sambles+=random_etiqueta
    random.shuffle(index_sambles)
    return index_sambles

def data_loader(data_set,subset):
    loader=DataLoader(Subset(data_set,subset),batch_size=32, shuffle=True)
    return loader



def data_for_clients(dic_labels=dict):
    test_perce=0.2
    number_client=len(dic_labels)
    dict_loaders={}
    for client in dic_labels.keys():
        labels=[] 
        samples_training=[]
        samples_testing=[]
        for i in range(4):
            label_i_amo=dic_labels[client][i]
            twenty=int(label_i_amo*test_perce)
            if label_i_amo!=0:
                labels.append(i)
                samples_training.append(label_i_amo-twenty)
                samples_testing.append(twenty)
        training_client=choosing_labels(train_set,labels,samples_training)
        testing_client=choosing_labels(test_set,labels,samples_testing)
        dict_loaders[client]=[data_loader(train_set,training_client),data_loader(test_set,testing_client)]
        #dict_loaders[client]=[labels,samples_training,samples_testing]
    return dict_loaders

def commun_test_set(dict):
        common=[]
        for client in dict.keys():
            local_loader=dict[client][1]
            common.append(local_loader.dataset)
        combined=ConcatDataset(common)
        combined_dataloader = DataLoader(combined, batch_size=32, shuffle=True)
        return combined_dataloader



if __name__ == "__main__":
    dic_labels={0:[100,100,0,0],1:[0,100,100,0],2:[0,0,0,100]}
    dic_loaders=data_for_clients(dic_labels)
    # common=commun_test_set(dic_loaders)
    common_three=len(dic_loaders[0][1])
    print(common_three)



