from .papcModel import NeuralNet as PAPC
from .papcModel import HyperParameters as HP


MODEL_NAME = 'PAPCNM'

class HyperParameters(HP):
    pass

class NeuralNet(PAPC):
    def __init__(self, systemParameters, grads):
        super(NeuralNet, self).__init__(systemParameters, grads)
        self.name = MODEL_NAME


    def forward(self, input):
        x, _ = input
        return super().forward([x, None])  # same as PAPC but without pilot information
