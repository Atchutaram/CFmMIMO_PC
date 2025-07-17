import torch.nn as nn

from .papcModel import NeuralNet as PAPC
from .papcModel import HyperParameters as HP
from .utils import EncoderLayer


MODEL_NAME = 'PAPCNM'

class HyperParameters(HP):
    pass

class NeuralNet(PAPC):
    def __init__(self, systemParameters, grads):
        super(NeuralNet, self).__init__(systemParameters, grads, HP=HyperParameters)
        self.name = MODEL_NAME
        self.layers = nn.ModuleList([
            EncoderLayer(self.M2, heads=self.heads, uniformAttention=True)
            for _ in range(self.num_layers)
        ])
