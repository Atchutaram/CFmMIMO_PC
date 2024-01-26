import torch
import torch.nn as nn

from .rootModel import CommonParameters, RootNet
from .utils import Norm
from powerControl.testing import project2s


MODEL_NAME = 'TDN'

# Hyper-parameters
class HyperParameters(CommonParameters):

    @classmethod
    def initialize(cls, simulationParameters, systemParameters):
        
        cls.preInt(simulationParameters, systemParameters)

        #  Room for any additional model-specific configurations
        cls.dropout = 0
        cls.inputSize = cls.K
        cls.outputSize = cls.K
        cls.numberOfAPs = cls.M
        
        if (simulationParameters.scenario == 0):
            cls.hiddenSize = int((1/(1-cls.dropout))*cls.K*1)
        elif (simulationParameters.scenario == 1):
            cls.hiddenSize = int((1/(1-cls.dropout))*cls.K*1)
        else:
            cls.hiddenSize = int((1/(1-cls.dropout))*cls.K*1)

        cls.outputShape = (-1, cls.M, cls.K)
        
    
    

class NeuralNet(RootNet):
    def __init__(self, systemParameters, grads):
        super(NeuralNet, self).__init__(systemParameters, grads)

        self.numSamples = HyperParameters.numSamples
        self.numEpochs = HyperParameters.numEpochs
        self.dataPath = HyperParameters.trainingDataPath
        self.valDataPath = HyperParameters.validationDataPath
        self.batchSize = HyperParameters.batchSize
        self.learningRate = HyperParameters.learningRate
        self.VARYING_STEP_SIZE = HyperParameters.VARYING_STEP_SIZE
        self.gamma = HyperParameters.gamma
        self.stepSize = HyperParameters.stepSize
        
        self.stepSize = HyperParameters.stepSize
        self.inputSize = HyperParameters.inputSize
        self.hiddenSize = HyperParameters.hiddenSize
        self.outputSize = HyperParameters.outputSize
        self.outputShape = HyperParameters.outputShape
        self.dropout = HyperParameters.dropout
        self.numberOfAPs = HyperParameters.numberOfAPs
        
        self.fcns = nn.ModuleList()
        for _ in range(self.numberOfAPs):
            self.fcns.append(
                nn.Sequential(
                    nn.Linear(self.inputSize, self.hiddenSize),
                    nn.ReLU(),
                    nn.Linear(self.hiddenSize, self.hiddenSize),
                    nn.ReLU(),
                    nn.Linear(self.hiddenSize, self.inputSize),
                    )
            )

        self.name = MODEL_NAME


    def forward(self, x):
        x, _ = x
        x = torch.unsqueeze(x, 1)
        output = []
        for m, FCN in enumerate(self.fcns):
            outputTemp = FCN(x[:, 0, m, :])
            
            outputTemp = torch.nn.functional.softplus(outputTemp + 6, beta = 2)
            outputTemp = torch.exp(-outputTemp)
            output.append(torch.unsqueeze(outputTemp, 0))
        
        output = torch.transpose(torch.cat(output), 0, 1)
        return output