import os
import torch
import torch.nn as nn
import importlib
import pickle
import math
import torch.nn.functional as F


from utils.utils import findTheLatestFile, findTheLatestFolder


findImportPath = lambda modelName : f"powerControl.models.{modelName.lower()}Model"

def initializeWeights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


def deploy(model, testSample, phiCrossMat, modelName, device):
    importPath = findImportPath(modelName)
    # module = importlib.import_module(importPath, ".")  # imports the scenarios
    
    

    with torch.no_grad():
        actualNumberOfUsers = phiCrossMat.size(-1)
        padUsers = model.maxNumberOfUsers - actualNumberOfUsers
        phiCrossMatPadded = F.pad(phiCrossMat, (0, padUsers, 0, padUsers), 'constant', 0)
        testSample = F.pad(testSample, (0, padUsers, 0, 0), 'constant', model.PAD_CONST)
        
        testSample = torch.log(testSample)
        testSampleShape = testSample.shape
        testSample = testSample.reshape((1,-1)).to(device='cpu')
        testSample.requires_grad=False
        testSample = testSample.to(device=device, dtype=torch.float32).view(testSampleShape)
        model.eval()
        model.to(device=device)
        
        mus_predicted = model(
                                    [
                                        testSample,
                                        phiCrossMatPadded.to(device=device, dtype=torch.float32)
                                    ]
                            )
        
        mus_predicted = mus_predicted[:,:,:actualNumberOfUsers]
        return mus_predicted

def initializeHyperParams(modelName, simulationParameters, systemParameters):
    # Ex: If model_name is 'ANN', it imports ANN_model module and initializes its hyper parameters.
    importPath = findImportPath(modelName)
    module = importlib.import_module(importPath, ".")  # imports the Models
    
    module.HyperParameters.initialize(simulationParameters, systemParameters)

def loadTheLatestModelAndParamsIfExists(
                                            modelName,
                                            modelFolder,
                                            systemParameters,
                                            grads,
                                            isTesting=False
                                        ):
    # For Training mode, the function first imports the appropriate model and initializes weights
    importPath = findImportPath(modelName)
    module = importlib.import_module(importPath, ".")  # imports the Models
        
    model = module.NeuralNet(systemParameters, grads)
    model.apply(initializeWeights)
    if isTesting:
        modelFolder = os.path.join(modelFolder, findTheLatestFolder(modelFolder), 'checkpoints')
        modelFile = findTheLatestFile(modelFolder)
    
        if modelFile:
            modelFilePath = os.path.join(modelFolder, modelFile)
            print('loading from: ', modelFilePath)
            model = module.NeuralNet.load_from_checkpoint(
                                                                modelFilePath,
                                                                system_parameters=systemParameters,
                                                                grads=grads
                                                        )
        else:
            from sys import exit
            print({modelFolder})
            print(f'Train the neural network before testing!')
            exit()
    
    return model

def get_nu_tensor(betas, systemParameters):
    from powerControl.utils import compute_v_mat
    phiCrossMat = systemParameters.phiCrossMat
    phiCrossMat = phiCrossMat.to(betas.device)
    
    # Eq (5) b X M X K
    vMat = compute_v_mat(betas, systemParameters.zeta_p, systemParameters.Tp, phiCrossMat)
    # Eq (14) b X M X K X K
    nuMat = torch.einsum('ik, mi, mk -> mki', phiCrossMat, (torch.sqrt(vMat) / betas), betas)

    return nuMat
    
def attention(query, key, value, d_k, mask=None, dropout=None):

    # query, key, value tensor are of dimension B x h x K x d_k
    # mask tensor is of dimension K x K
    
    scores = (query @ key.transpose(-2, -1)) /  math.sqrt(d_k)

    if mask is not None:
        scores = scores*mask
    
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
        
    output = scores @ value
    
    return output  # dimension B x h x K x d_k

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, M, dropout = 0.1):
        super().__init__()
        
        self.M = M
        self.d_k = M // heads
        if M % heads:
            from sys import exit
            print(f('The number of APs (= {M}) is not a multiple of the number of heads (={heads}).'
                    'Re-adjust the number of heads.'))
            exit()

        self.h = heads
        
        self.qLinear = nn.Linear(M, M)
        self.vLinear = nn.Linear(M, M)
        self.kLinear = nn.Linear(M, M)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(M, M)
    
    def forward(self, x, mask=None):
        
        # x is of dimension B x K x M
        B, _, _ = x.shape

        
        # perform linear operation and split into h heads

        query = self.qLinear(x).view(B, -1, self.h, self.d_k)
        key = self.kLinear(x).view(B, -1, self.h, self.d_k)
        value = self.vLinear(x).view(B, -1, self.h, self.d_k)
        
        # transpose to get dimensions B x h x K x d_k
       
        query = query.transpose(1,2)
        key = key.transpose(1,2)
        value = value.transpose(1,2)

        scores = attention(query, key, value, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(B, -1, self.M)

        output = self.out(concat)
        return output  # dimensions B x K x M


class FeedForward(nn.Module):

    def __init__(self, M, dropout = 0.1):
        super().__init__() 
        
        dMid = int(1 / (1-dropout)) * M

        self.linear1 = nn.Linear(M, dMid)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dMid, M)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.linear2(x)
        return x


class Norm(nn.Module):
    
    def __init__(self, size, eps = 1e-6):
        super().__init__()
    
        self.size = size
        # create two learnable parameters to calibrate normalization
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    
    def forward(self, x):
        den = (x.std(dim=[-2, -1], keepdim=True) + self.eps)
        norm = self.alpha * (x - x.mean(dim=[-2, -1], keepdim=True)) / den + self.bias
        return norm

class EncoderLayer(nn.Module):
    
    def __init__(self, M, heads, dropout=0.1):
        super().__init__()
        self.norm1 = Norm(M)
        self.norm2 = Norm(M)
        self.attn = MultiHeadAttention(heads, M, dropout)
        self.ff = FeedForward(M, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x = x + self.dropout1(self.attn(x, mask))
        x = self.norm1(x)
        x = x + self.dropout2(self.ff(x))
        x = self.norm2(x)
        return x