# -*- coding: utf-8 -*-
import os
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
import numpy as np
import pandas as pd
from parameters import *
from normalization import Normalization
from loadDataset import *
from model import *
from errorAnalysis import *
from time import time

if __name__ == '__main__':

    torch.manual_seed(0)
    os.system('mkdir models')
    os.system('mkdir loss-history')
    
    ##############---FWD MODEL---##############
    fwdModel = createFNN(featureDim, fwdHiddenDim, fwdHiddenLayers, labelDim)
    fwdOptimizer = torch.optim.Adam(fwdModel.parameters(), lr=fwdLearningRate)    
    print('\n\n**************************************************************')
    print('fwdModel', fwdModel)
    print('**************************************************************\n')

    ##############---INV MODEL---##############
    invModel = createINN(labelDim, invHiddenDim, invHiddenLayers, featureDim)
    invOptimizer = torch.optim.Adam(invModel.parameters(), lr=invLearningRate)
    print('\n\n**************************************************************')
    print('invModel', invModel)
    print('**************************************************************\n')

    ##############---INIT DATA---##############
    train_set, valid_set, test_set, featureNormalization, labelNormalization = getDataset()
    train_data_loader = DataLoader(dataset=train_set, num_workers=numWorkers, batch_size=batchSize, shuffle=batchShuffle)
    valid_data_loader = DataLoader(dataset=valid_set, num_workers=numWorkers, batch_size=batchSize, shuffle=False)
    test_data_loader = DataLoader(dataset=test_set, num_workers=numWorkers, batch_size=batchSize, shuffle=False)

    ##############---Training---##############
    fwdEpochLoss = 0.0
    invEpochLoss = 0.0

    fwdTrainHistory = []
    fwdValidHistory = []
    invTrainHistory = []
    invValidHistory = []
    loader_all_train = DataLoader(dataset=train_set, num_workers=numWorkers, batch_size=len(train_set), shuffle=False)
    loader_all_valid = DataLoader(dataset=valid_set, num_workers=numWorkers, batch_size=len(valid_set), shuffle=False)
    loader_all_test = DataLoader(dataset=test_set, num_workers=numWorkers, batch_size=len(test_set), shuffle=False)
    x_all_train, y_all_train = next(iter(loader_all_train))
    x_all_valid, y_all_valid = next(iter(loader_all_valid))
    x_all_test, y_all_test = next(iter(loader_all_test))
   
    if(fwdTrain):
        fwdBestLoss = 1e6
        fwdValidLoss = 1e6
        patienceCount = 0
        patienceTrigger = False
        print('\nBeginning forward model training')
        print('-------------------------------------')
        ##############---FWD TRAINING---##############
        for fwdEpochIter in range(fwdEpochs):
            # ---------------------------------------
            #####  Early stopping with patience
            if fwdValidLoss >= fwdBestLoss:
                patienceCount += 1
                if patienceCount >= fwdPatience:
                    print('-------------------------------------')
                    print('Early stopping the training')
                    print('-------------------------------------')
                    patienceTrigger = True
            else:
                patienceCount = 0
                fwdBestLoss = fwdValidLoss
            
            if patienceTrigger == True:
                break
            # ---------------------------------------
            fwdEpochLoss = 0.0
            for iteration, batch in enumerate(train_data_loader, 0):
                #get batch
                x_train = batch[0]
                y_train = batch[1]
                #set train mode
                fwdModel.train()
                #predict
                y_train_pred = fwdModel(x_train)
                #compute loss
                y_train_pred_SPD = fwdSPDTransform(y_train_pred)
                
                fwdLoss = fwdLossFn(labelNormalization.normalize(y_train_pred_SPD), labelNormalization.normalize(y_train))
                #optimize
                fwdOptimizer.zero_grad()
                fwdLoss.backward()
                fwdOptimizer.step()
                #store loss
                fwdEpochLoss += fwdLoss.item()

            print(" {}:{}/{} | fwdEpochLoss: {:.2e} | invEpochLoss: {:.2e} | PatienceCount {:.1f}".format(\
                "fwd",fwdEpochIter,fwdEpochs,fwdEpochLoss/len(train_data_loader),invEpochLoss/len(train_data_loader), patienceCount))

            fwdModel.eval()
            fwdTrainLoss = fwdLossFn(labelNormalization.normalize(fwdSPDTransform(fwdModel(x_train.clone()))),labelNormalization.normalize(y_train.clone())).item()
            fwdValidLoss = fwdLossFn(labelNormalization.normalize(fwdSPDTransform(fwdModel(x_all_valid.clone()))),labelNormalization.normalize(y_all_valid.clone())).item()
            fwdTrainHistory.append(fwdTrainLoss)
            fwdValidHistory.append(fwdValidLoss)

        print('-------------------------------------')
        #save model
        torch.save(fwdModel, "models/fwdModel.pt")
        #export loss history
        exportList('loss-history/fwdTrainHistory',fwdTrainHistory)
        exportList('loss-history/fwdValidHistory',fwdValidHistory)
    else:
        fwdModel = torch.load("models/fwdModel.pt")
        fwdModel.eval()

    if(invTrain):
        invBestLoss = 1e6
        invValidLoss = 1e6
        patienceCount = 0
        patienceTrigger = False
        print('\nBeginning inverse model training')
        print('-------------------------------------')
        ##############---INV TRAINING---##############
        for invEpochIter in range(invEpochs):
            # ---------------------------------------
            #####  Early stopping with patience
            if invValidLoss >= (invBestLoss-invPatienceTol):
                patienceCount += 1
                if patienceCount >= invPatience:
                    print('-------------------------------------')
                    print('Early stopping the training')
                    print('-------------------------------------')
                    patienceTrigger = True
            else:
                patienceCount = 0
                invBestLoss = invValidLoss
            # ---------------------------------------
            invEpochLoss = 0.0        
            for iteration, batch in enumerate(train_data_loader, 0):
                #get batch
                x_train = batch[0]
                y_train = batch[1]
                #set train mode
                invModel.train()
                #predict
                x_train_pred = invModel(y_train)
                y_train_pred_pred = fwdModel(x_train_pred)
                y_train_pred_pred_SPD = fwdSPDTransform(y_train_pred_pred)
                #compute loss
                invLoss =  invLossFn(y_train_pred_pred_SPD, y_train)
                #optimize
                invOptimizer.zero_grad()
                invLoss.backward()
                invOptimizer.step()
                #store loss
                invEpochLoss += invLoss.item()
            print(" {}:{}/{} | fwd EpochLoss: {:.2e} | invEpochLoss: {:.6e} | PatienceCount {:.1f}".format(\
                "inv",invEpochIter,invEpochs, fwdEpochLoss/len(train_data_loader),invEpochLoss/len(train_data_loader), patienceCount))
            invTrainLoss = invLossFn(fwdSPDTransform(fwdModel(invModel(y_train.clone()))),y_train.clone()).item()
            invValidLoss = invLossFn(fwdSPDTransform(fwdModel(invModel(y_all_valid.clone()))),y_all_valid.clone()).item()
            invTrainHistory.append(invTrainLoss)
            invValidHistory.append(invValidLoss)

            if patienceTrigger == True:
                break

        print('-------------------------------------')
        #save model
        torch.save(invModel, "models/invModel.pt")
        #export loss history
        exportList('loss-history/invTrainHistory',invTrainHistory)
        exportList('loss-history/invValidHistory',invValidHistory)

    else:
        invModel = torch.load("models/invModel.pt")
        invModel.eval()

    #############---TESTING---##############
    x_test, y_test = next(iter(test_data_loader))
    
    with torch.no_grad():
        y_test_pred = fwdModel(x_test)
        y_test_pred_SPD = fwdSPDTransform(y_test_pred)
        x_test_pred = invModel(y_test)
        y_test_pred_pred = fwdModel(x_test_pred)
        y_test_pred_pred_SPD = fwdSPDTransform(y_test_pred_pred)
    
        #############---POST PROC---##############
        print('\nR2 values:\n--------------------------------------------')
        print('Fwd test Y R2:',computeR2(y_test_pred_SPD, y_test),'\n')
        print('Inv test reconstruction Y R2:',computeR2(y_test_pred_pred_SPD, y_test),'\n')

