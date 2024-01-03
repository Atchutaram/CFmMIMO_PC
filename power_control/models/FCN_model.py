import torch
import torch.nn as nn
import os
from sklearn.preprocessing import StandardScaler

from .root_model import CommonParameters, RootNet
from .utils import Norm
from power_control.testing import project_to_s


MODEL_NAME = 'FCN'

# Hyper-parameters
class HyperParameters(CommonParameters):

    @classmethod
    def intialize(cls, simulation_parameters, system_parameters):
        
        cls.pre_int(simulation_parameters, system_parameters)

        #  Room for any additional model-specific configurations
        cls.dropout = 0
        cls.input_size = cls.M * cls.K
        cls.output_size = cls.M * cls.K
        MK = 1 * cls.M * cls.K
        
        if (simulation_parameters.scenario == 0):
            cls.hidden_size = int((1/(1-cls.dropout))*MK*4)
        elif (simulation_parameters.scenario == 1):
            cls.hidden_size = int((1/(1-cls.dropout))*MK/2)
        else:
            cls.hidden_size = int((1/(1-cls.dropout))*MK/7)

        cls.output_shape = (-1, cls.M, cls.K)
        

class NeuralNet(RootNet):
    def __init__(self, system_parameters, grads):
        super(NeuralNet, self).__init__(system_parameters, grads)

        self.n_samples = HyperParameters.n_samples
        self.num_epochs = HyperParameters.num_epochs
        self.data_path = HyperParameters.training_data_path
        self.val_data_path = HyperParameters.validation_data_path
        self.batch_size = HyperParameters.batch_size
        self.learning_rate = HyperParameters.learning_rate
        self.VARYING_STEP_SIZE = HyperParameters.VARYING_STEP_SIZE
        self.gamma = HyperParameters.gamma
        self.step_size = HyperParameters.step_size
        
        self.step_size = HyperParameters.step_size
        self.input_size = HyperParameters.input_size
        self.hidden_size = HyperParameters.hidden_size
        self.output_size = HyperParameters.output_size
        self.output_shape = HyperParameters.output_shape
        self.dropout = HyperParameters.dropout

        self.hidden = lambda: nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            Norm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
        )
        
        self.FCN_full = nn.Sequential(
            Norm(self.input_size),
            nn.Linear(self.input_size, self.hidden_size),
            Norm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            self.hidden(),
            nn.Linear(self.hidden_size, self.output_size),
            Norm(self.output_size),
        )
        
        self.name = MODEL_NAME


    def forward(self, x):
        x, _ = x
        output = self.FCN_full(x.view(-1, 1, self.input_size))
        output = output.view(self.output_shape)
        
        output = torch.nn.functional.softplus(output+6, beta = 2)
        
        y = torch.exp(-output)
        
        output = project_to_s(y, self.N_inv_root)
        
        return output
