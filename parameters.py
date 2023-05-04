import torch

torch.manual_seed(123)

dataset_dir = 'data/'

featureDim = 14
labelDim = 6

trainSplit = 0.8
validSplit = 0.15
testSplit = 0.05/(1-trainSplit)

batchSize = 4096
batchShuffle = True
numWorkers = 0

randomWeights = False

fwdTrain = False
fwdEpochs =  1000
fwdPatience = 50
fwdHiddenDim = [512, 512, 512, 512, 512]
fwdHiddenLayers = len(fwdHiddenDim)-1
fwdLearningRate = 1e-3
fwdLossFn = torch.nn.MSELoss()

invTrain = True
invEpochs =  1000
invPatience = 50
invPatienceTol = 1e-6
invHiddenDim = [512, 512, 512, 512, 512]
invHiddenLayers = len(invHiddenDim)-1
invLearningRate = 1e-3
invLossFn = torch.nn.MSELoss()