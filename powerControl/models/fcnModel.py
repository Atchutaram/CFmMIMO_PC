import torch
import torch.nn as nn

from .rootModel import CommonParameters, RootNet
from .utils import Norm
from powerControl.testing import project2s


MODEL_NAME = 'FCN'

# Hyper-parameters
class HyperParameters(CommonParameters):

    @classmethod
    def initialize(cls, simulationParameters, systemParameters):
        
        cls.preInt(simulationParameters, systemParameters)

        #  Room for any additional model-specific configurations
        cls.inputSize = cls.M * cls.K
        cls.outputSize = cls.M * cls.K
        MK = 1 * cls.M * cls.K
        
        if (simulationParameters.scenario == 0):
            cls.hiddenSize = MK*4
            cls.learningRate = 1e-1
        elif (simulationParameters.scenario == 1):
            cls.hiddenSize = int(MK/2)
        elif (simulationParameters.scenario == 2):
            cls.hiddenSize = int(MK/7)
        elif (simulationParameters.scenario == 3):
            cls.hiddenSize = int(MK/28)
        else:
            raise('Invalid Scenario Configuration')

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
        self.lambdaLr = HyperParameters.lambdaLr
        
        self.inputSize = HyperParameters.inputSize
        self.hiddenSize = HyperParameters.hiddenSize
        self.outputSize = HyperParameters.outputSize
        self.outputShape = HyperParameters.outputShape

        self.hidden = lambda: nn.Sequential(
            nn.Linear(self.hiddenSize, self.hiddenSize),
            Norm(self.hiddenSize),
            nn.ReLU(),
        )
        
        self.fcnFull = nn.Sequential(
            Norm(self.inputSize),
            nn.Linear(self.inputSize, self.hiddenSize),
            Norm(self.hiddenSize),
            nn.ReLU(),
            self.hidden(),
            nn.Linear(self.hiddenSize, self.outputSize),
            Norm(self.outputSize),
        )
        
        self.name = MODEL_NAME


    def forward(self, x):
        x, _ = x
        output = self.fcnFull(x.view(-1, 1, self.inputSize))
        output = output.view(self.outputShape)
        
        output = torch.nn.functional.softplus(output + 6, beta = 2)
        
        y = torch.exp(-output)
        
        output = project2s(y, self.N_invRoot)
        
        return output
