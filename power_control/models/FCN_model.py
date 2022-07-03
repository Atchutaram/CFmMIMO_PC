import torch
import torch.nn as nn
import os
from sklearn.preprocessing import StandardScaler

from .root_model import RootDataset, CommonParameters, RootNet
from .utils import Norm


MODEL_NAME = 'FCN'

class BetaDataset(RootDataset):
    def __init__(self, data_path, phi_orth, normalizer, mode, n_samples):
        self.path = data_path
        _, _, files = next(os.walk(self.path))
        self.n_samples = min(len(list(filter(lambda k: 'betas' in k, files))), n_samples)
        self.sc = normalizer
        self.mode = mode
        self.phi_orth = phi_orth
        
    def __getitem__(self, index):
        beta_file_name = f'betas_sample{index}.pt'
        beta_file_path = os.path.join(self.path, beta_file_name)
        m = torch.load(beta_file_path)
        beta_original = m['betas'].to(dtype=torch.float32)
        pilot_sequence = m['pilot_sequence'].to(dtype=torch.int32)

        phi = torch.index_select(self.phi_orth.to(dtype=torch.float32), 0, pilot_sequence)
        phi_cross_mat = torch.abs(phi.conj() @ phi.T)
        
        beta_torch = torch.log(beta_original)
        beta_torch = beta_torch.to(dtype=torch.float32)
        beta_torch = beta_torch.reshape(beta_original.shape)

        return phi_cross_mat, beta_torch, beta_original

    def __len__(self):
        return self.n_samples

# Hyper-parameters
class HyperParameters(CommonParameters):
    sc = StandardScaler()
    sc_path = os.path.join(os.getcwd(), f'{MODEL_NAME}_sc.pkl')
    training_data_path = ''
    InpDataSet = BetaDataset

    @classmethod
    def intialize(cls, simulation_parameters, system_parameters, is_test_mode):
        
        cls.pre_int(simulation_parameters, system_parameters)
        cls.dropout = 0.5
        cls.input_size = cls.M * cls.K
        cls.output_size = cls.M * cls.K
        MK = 1 * cls.M * cls.K
        if 0 <= cls.dropout < 1:
            cls.hidden_size = int((1/(1-cls.dropout))*MK)
        else:
            cls.hidden_size = int(MK)

        cls.output_shape = (-1, cls.M, cls.K)
        
        if is_test_mode:
            cls.batch_size = 1
            return

        if cls.scenario == 0:
            cls.batch_size = 8 * 2
        elif cls.scenario == 1:
            cls.batch_size = 8 * 2
        else:
            cls.batch_size = 1
    

class NeuralNet(RootNet):
    def __init__(self, system_parameters, grads):
        super(NeuralNet, self).__init__(system_parameters, grads)
        
        self.n_samples = HyperParameters.n_samples
        self.num_epochs = HyperParameters.num_epochs
        self.eta = HyperParameters.eta
        self.data_path = HyperParameters.training_data_path
        self.val_data_path = HyperParameters.validation_data_path
        self.normalizer = HyperParameters.sc
        self.batch_size = HyperParameters.batch_size
        self.learning_rate = HyperParameters.learning_rate
        self.VARYING_STEP_SIZE = HyperParameters.VARYING_STEP_SIZE
        self.gamma = HyperParameters.gamma
        self.step_size = HyperParameters.step_size
        self.input_size = HyperParameters.input_size
        self.hidden_size = HyperParameters.hidden_size
        self.output_size = HyperParameters.output_size
        self.output_shape = HyperParameters.output_shape
        self.InpDataset = HyperParameters.InpDataSet
        self.dropout = HyperParameters.dropout
        
        self.hidden_layer_1 = nn.Sequential(
            Norm(self.input_size),
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
        )
        
        
        self.hidden_layer_2 = nn.Sequential(

            Norm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
        )
        
        
        self.output_layer = nn.Sequential(
            Norm(self.hidden_size),
            nn.Linear(self.hidden_size, self.output_size),
            nn.Sigmoid(),
        )
        
        self.name = MODEL_NAME


    def forward(self, x):
        x, _ = x
        output = self.hidden_layer_1(x.view(-1, 1, self.input_size))
        output = self.hidden_layer_2(output)
        output = self.output_layer(output)
        output = output.view(self.output_shape)*1e-1
        return output