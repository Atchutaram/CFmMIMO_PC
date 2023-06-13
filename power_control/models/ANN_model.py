import torch
import torch.nn as nn


from .root_model import CommonParameters, RootNet
from .utils import EncoderLayer, Norm
from power_control.testing import project_to_s


MODEL_NAME = 'ANN'

# Hyper-parameters
class HyperParameters(CommonParameters):

    @classmethod
    def intialize(cls, simulation_parameters, system_parameters):
        
        cls.pre_int(simulation_parameters, system_parameters)

        #  Room for any additional model-specific configurations
        cls.heads = 5
        if (simulation_parameters.scenario == 0):
            M2 = 16*cls.heads
        else:
            M2 = 40*cls.heads
        cls.M2 = int(1 / (1-cls.dropout))*M2

    

class NeuralNet(RootNet):
    def __init__(self, system_parameters, grads):
        super(NeuralNet, self).__init__(system_parameters, grads)
        M = HyperParameters.M
        self.n_samples = HyperParameters.n_samples
        self.num_epochs = HyperParameters.num_epochs
        self.data_path = HyperParameters.training_data_path
        self.val_data_path = HyperParameters.validation_data_path
        self.batch_size = HyperParameters.batch_size
        self.learning_rate = HyperParameters.learning_rate
        self.VARYING_STEP_SIZE = HyperParameters.VARYING_STEP_SIZE
        self.gamma = HyperParameters.gamma
        self.step_size = HyperParameters.step_size
        
        dropout = HyperParameters.dropout
        heads = HyperParameters.heads
        M2 = HyperParameters.M2

        self.norm1 = Norm(M)
        self.inp_mapping = nn.Linear(M, M2)
        self.norm2 = Norm(M2)
        self.layer1 = EncoderLayer(M2, heads=heads, dropout=dropout)
        self.layer2 = EncoderLayer(M2, heads=heads, dropout=dropout)
        self.layer3 = EncoderLayer(M2, heads=heads, dropout=dropout)
        # self.layer4 = EncoderLayer(M2, heads=heads, dropout=dropout)
        # self.layer5 = EncoderLayer(M2, heads=heads, dropout=dropout)
        # self.layer6 = EncoderLayer(M2, heads=heads, dropout=dropout)
        self.otp_mapping = nn.Linear(M2, M)
        self.norm3 = Norm(M)

        self.name = MODEL_NAME


    def forward(self, input):
        x, phi_cross_mat = input
        mask = torch.unsqueeze(phi_cross_mat**2, dim=1)
        x = x.transpose(1,2).contiguous()
        x = self.norm1(x)
        x = self.inp_mapping(x)
        
        x = self.norm2(x)
        x = self.layer1(x, mask=mask)
        x = self.layer2(x, mask=mask)
        x = self.layer3(x, mask=mask)
        # x = self.layer4(x, mask=mask)
        # x = self.layer5(x, mask=mask)
        # x = self.layer6(x, mask=mask)
        
        x = self.otp_mapping(x)
        x = self.norm3(x)
        x = (x.transpose(1,2).contiguous()+6)
        x = torch.nn.functional.softplus(x, beta = 2)
        
        y = torch.exp(-x)
        
        output = project_to_s(y, self.N_inv_root)
        
        return output