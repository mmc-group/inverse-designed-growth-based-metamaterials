import torch
from parameters import *
import torch.nn.functional as F


def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)

def createFNN(inputDim,hiddenDim,layers,outputDim):
    model = torch.nn.Sequential()
    model.add_module('Fin',torch.nn.Linear(inputDim, hiddenDim[0]))
    for i in range(layers):
        model.add_module('F'+str(i+1),torch.nn.Linear(hiddenDim[i],hiddenDim[i+1]))
        model.add_module('SoftPlus'+str(i+1),torch.nn.Softplus())
    model.add_module('Fout',torch.nn.Linear(hiddenDim[layers],outputDim))
    if(randomWeights):
        model.apply(weights_init_uniform_rule)
    return model

def fwdSPDTransform(y_pred):

    L = torch.zeros((y_pred.size(0),3,3))
    L[:,0,0] = torch.nn.functional.softplus(y_pred[:,0].clone())
    L[:,1,0] = y_pred[:,1].clone()
    L[:,2,0] = y_pred[:,2].clone()
    L[:,1,1] = torch.nn.functional.softplus(y_pred[:,3].clone())
    L[:,2,1] = y_pred[:,4].clone()
    L[:,2,2] = torch.nn.functional.softplus(y_pred[:,5].clone())
    y_pred[:,0] = L[:,0,0]*L[:,0,0]
    y_pred[:,1] = L[:,0,0]*L[:,1,0]
    y_pred[:,2] = L[:,0,0]*L[:,2,0]
    y_pred[:,3] = L[:,1,0]*L[:,1,0] + L[:,1,1]*L[:,1,1]
    y_pred[:,4] = L[:,1,0]*L[:,2,0] + L[:,1,1]*L[:,2,1]
    y_pred[:,5] = L[:,2,0]*L[:,2,0] + L[:,2,1]*L[:,2,1] + L[:,2,2]*L[:,2,2]

    return y_pred

def createINN(inputDim,hiddenDim,layers,outputDim):
    model = torch.nn.Sequential()
    model.add_module('Fin',torch.nn.Linear(inputDim, hiddenDim[0]))
    for i in range(layers):
        model.add_module('F'+str(i+1),torch.nn.Linear(hiddenDim[i],hiddenDim[i+1]))
        model.add_module('Softplus'+str(i+1),torch.nn.Softplus())
    model.add_module('Fout',torch.nn.Linear(hiddenDim[layers],outputDim))
    model.add_module('SigmoidOut',torch.nn.Sigmoid())
    if(randomWeights):
        model.apply(weights_init_uniform_rule)
    return model