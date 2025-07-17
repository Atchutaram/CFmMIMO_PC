import torch
import torch.nn as nn


from .rootModel import CommonParameters, RootNet
from .utils import EncoderLayer, Norm
from powerControl.testing import project2s


MODEL_NAME = 'PAPC'

# Hyper-parameters
class HyperParameters(CommonParameters):

    @classmethod
    def initialize(cls, simulationParameters, systemParameters):
        cls.preInt(simulationParameters, systemParameters)

        #  Room for any additional model-specific configurations
        cls.heads = 5
        if (simulationParameters.scenario == 0):
            cls.M2 = 8 * cls.heads
        elif (
                    (simulationParameters.scenario == 1) or
                    (simulationParameters.scenario == 2) or
                    (simulationParameters.scenario == 3)
            ):
            cls.M2 = cls.M2Multiplier * cls.heads
        else:
            raise('Invalid Scenario Configuration')
    

class NeuralNet(RootNet):
    def __init__(self, systemParameters, grads, HP=None):
        super(NeuralNet, self).__init__(systemParameters, grads)
        if HP is None:
            HP = HyperParameters
        
        M = HP.M
        self.numSamples = HP.numSamples
        self.numEpochs = HP.numEpochs
        self.dataPath = HP.trainingDataPath
        self.valDataPath = HP.validationDataPath
        self.batchSize = HP.batchSize
        self.learningRate = HP.learningRate
        self.VARYING_STEP_SIZE = HP.VARYING_STEP_SIZE
        self.lambdaLr = HP.lambdaLr
        
        self.heads = HP.heads
        self.M2 = HP.M2

        self.norm1 = Norm(M)
        self.inpMapping = nn.Linear(M, self.M2)
        self.norm2 = Norm(self.M2)
        self.num_layers = 3
        self.layers = nn.ModuleList([
            EncoderLayer(self.M2, heads=self.heads) for _ in range(self.num_layers)
        ])
        self.otpMapping = nn.Linear(self.M2, M)
        self.norm3 = Norm(M)

        self.name = MODEL_NAME


    def forward(self, input):
        x, phiCrossMat = input
        mask = None
        if phiCrossMat is not None:
            mask = torch.unsqueeze(phiCrossMat ** 2, dim=1)
        x = x.transpose(1,2).contiguous()
        x = self.norm1(x)
        x = self.inpMapping(x)
        
        x = self.norm2(x)
        for layer in self.layers:
            x = layer(x, mask=mask)
        
        x = self.otpMapping(x)
        x = self.norm3(x)
        x = torch.nn.functional.relu(x.transpose(1,2).contiguous() + 6)
        
        y = torch.exp(-x)
        output = project2s(y, self.N_invRoot)
        
        return output
    
    def dumpAttention(self, input, dumpPath):
        x, phiCrossMat = input
        mask = None
        if phiCrossMat is not None:
            mask = torch.unsqueeze(phiCrossMat ** 2, dim=1)
        x = x.transpose(1,2).contiguous()
        x = self.norm1(x)
        x = self.inpMapping(x)
        
        x = self.norm2(x)
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                x = layer(x, mask=mask, dumpPath=dumpPath)
            else:
                x = layer(x, mask=mask)

        
        x = self.otpMapping(x)
        x = self.norm3(x)
        x = torch.nn.functional.relu(x.transpose(1,2).contiguous() + 6)
        
        y = torch.exp(-x)
        output = project2s(y, self.N_invRoot)
        
        return output
    