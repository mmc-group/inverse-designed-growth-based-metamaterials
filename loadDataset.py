# -*- coding: utf-8 -*-
import torch
import numpy as np
import pandas as pd
from parameters import *
from normalization import Normalization
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, TensorDataset, DataLoader

def getDataset():
    
    ######################################################
    dir = dataset_dir
    ######################################################
    
    featureTensor = torch.tensor(torch.load(dir + 'X_data.pth'))
    labelTensor = torch.tensor(torch.load(dir + 'y_data.pth'))


    ######################################################
    
    print('Input data: ',featureTensor.shape)          
    print('Output data: ',labelTensor.shape)          

    ##############---INIT NORMALIZATION---##############
    featureNormalization = Normalization(featureTensor)
    featureTensor = featureNormalization.normalize(featureTensor)
    labelNormalization = Normalization(labelTensor)
    
    ##############---INIT Dataset and loader---##############
    dataset =  TensorDataset(featureTensor.float(), labelTensor.float())
    l1 = round(len(dataset)*trainSplit)
    l2 = round(len(dataset)*validSplit)
    l3 = len(dataset) - l1 - l2
    print('train/valid/test: ',[l1,l2,l3])
    train_set, valid_test_set = torch.utils.data.random_split(dataset, [l1,l2+l3])
    valid_set, test_set = torch.utils.data.random_split(valid_test_set,[l2,l3])
    
    return train_set, valid_set, test_set, featureNormalization, labelNormalization


#################################################     
def exportTensor(name,data,cols, header=True):
    df=pd.DataFrame.from_records(data.detach().numpy())
    if(header):
        df.columns = cols
    print(name)
    df.to_csv(name+".csv",header=header)

def exportList(name,data):
    arr=np.array(data)
    np.savetxt(name+".csv", [arr], delimiter=',')
    
    