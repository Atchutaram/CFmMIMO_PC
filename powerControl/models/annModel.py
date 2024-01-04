import torch
import torch.nn as nn


from .rootModel import CommonParameters, RootNet
from .utils import EncoderLayer, Norm
from powerControl.testing import project2s


MODEL_NAME = 'ANN'

# Hyper-parameters
class HyperParameters(CommonParameters):

    @classmethod
    def intialize(cls, simulationParameters, systemParameters):
        
        cls.preInt(simulationParameters, systemParameters)

        #  Room for any additional model-specific configurations
        cls.heads = 5
        if (simulationParameters.scenario == 0):
            M2 = 16*cls.heads
        else:
            M2 = 200*cls.heads
        cls.M2 = int(1 / (1-cls.dropout))*M2

    

class NeuralNet(RootNet):
    def __init__(self, systemParameters, grads):
        super(NeuralNet, self).__init__(systemParameters, grads)
        M = HyperParameters.M
        self.numSamples = HyperParameters.numSamples
        self.numEpochs = HyperParameters.numEpochs
        self.dataPath = HyperParameters.trainingDataPath
        self.valDataPath = HyperParameters.validationDataPath
        self.batchSize = HyperParameters.batchSize
        self.learningRate = HyperParameters.learningRate
        self.VARYING_STEP_SIZE = HyperParameters.VARYING_STEP_SIZE
        self.gamma = HyperParameters.gamma
        self.stepSize = HyperParameters.stepSize
        
        dropout = HyperParameters.dropout
        heads = HyperParameters.heads
        M2 = HyperParameters.M2

        self.norm1 = Norm(M)
        self.inpMapping = nn.Linear(M, M2)
        self.norm2 = Norm(M2)
        self.layer1 = EncoderLayer(M2, heads=heads, dropout=dropout)
        self.layer2 = EncoderLayer(M2, heads=heads, dropout=dropout)
        self.layer3 = EncoderLayer(M2, heads=heads, dropout=dropout)
        self.otpMapping = nn.Linear(M2, M)
        self.norm3 = Norm(M)

        self.name = MODEL_NAME


    def forward(self, input):
        x, phiCrossMat = input
        mask = torch.unsqueeze(phiCrossMat**2, dim=1)
        x = x.transpose(1,2).contiguous()
        x = self.norm1(x)
        x = self.inpMapping(x)
        
        x = self.norm2(x)
        x = self.layer1(x, mask=mask)
        x = self.layer2(x, mask=mask)
        x = self.layer3(x, mask=mask)
        
        x = self.otpMapping(x)
        x = self.norm3(x)
        x = (x.transpose(1,2).contiguous()+6)
        x = torch.nn.functional.softplus(x, beta = 2)
        
        y = torch.exp(-x)
        
        output = project2s(y, self.N_invRoot)
        
        return output
