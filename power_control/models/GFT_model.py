import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import pickle
from sklearn.preprocessing import StandardScaler

from .root_model import Mode, RootDataset, CommonParameters, RootNet


MODEL_NAME = 'GFT'

class BetaDataset(RootDataset):
    def __init__(self, data_path, normalizer, mode, n_samples, sqrt_laplace_matrix, device):
        super(BetaDataset, self).__init__(data_path, normalizer, mode, n_samples, device)
        self.sqrt_laplace_matrix = sqrt_laplace_matrix.to(torch.device('cpu'))

    def __getitem__(self, index):
        beta_file_name = f'betas_sample{index}.pt'
        beta_file_path = os.path.join(self.path, beta_file_name)
        beta_original = torch.load(beta_file_path)['betas'].to(dtype=torch.float32)
        beta_torch = self.sqrt_laplace_matrix @ beta_original
        if self.mode == Mode.pre_processing:
            beta_torch = beta_torch.reshape((-1,))
            return beta_torch
        
        beta_torch = beta_torch.reshape((1, -1,))
        beta_torch = self.sc.transform(beta_torch)[0]
        beta_torch = torch.from_numpy(beta_torch).to(dtype=torch.float32, device=self.device)
        beta_torch = beta_torch.reshape(beta_original.shape)
        return beta_torch, beta_original.to(device=self.device)


# Hyper-parameters
class HyperParameters(CommonParameters):
    sc = StandardScaler()
    sc_path = os.path.join(os.getcwd(), f'{MODEL_NAME}_sc.pkl')
    training_data_path = ''
    
    @classmethod
    def intialize(cls, simulation_parameters, system_parameters, is_test_mode):
        M = system_parameters.number_of_access_points
        K = system_parameters.number_of_users
        cls.sqrt_laplace_matrix = system_parameters.sqrt_laplace_matrix

        cls.input_size = K * M
        cls.output_size = K * M
        cls.hidden_size = M * K
        cls.output_shape = (-1, M, K)

        
        cls.n_samples = simulation_parameters.number_of_samples
        cls.training_data_path = simulation_parameters.data_folder
        cls.scenario = simulation_parameters.scenario
        
        if is_test_mode:
            cls.batch_size = 1
            return

        if cls.scenario == 1:
            cls.batch_size = 8 * 2
        else:
            cls.batch_size = 1
        

        train_dataset = BetaDataset(data_path=cls.training_data_path, normalizer=cls.sc, mode=Mode.pre_processing, n_samples=cls.n_samples, device=torch.device('cpu'), sqrt_laplace_matrix = cls.sqrt_laplace_matrix)
        train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False)
        
        for beta in train_loader:
            with torch.no_grad():
                cls.sc.partial_fit(beta)

        pickle.dump(cls.sc, open(cls.sc_path, 'wb'))  # saving sc for loading later in testing phase
        print(f'{cls.sc_path} dumped!')
    
    

class NeuralNet(RootNet):
    def __init__(self, device, system_parameters, interm_folder, grads):
        super(NeuralNet, self).__init__(device, system_parameters, interm_folder, grads)
        
        self.n_samples = HyperParameters.n_samples
        self.num_epochs = HyperParameters.num_epochs
        self.eta = HyperParameters.eta
        self.data_path = HyperParameters.training_data_path
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
        
        self.FCN = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size),
            nn.ReLU(),
        )
        self.sqrt_laplace_matrix = system_parameters.sqrt_laplace_matrix
        
        self.name = MODEL_NAME
        self.to(self.device)
        self.InpDataset = BetaDataset

    def forward(self, x):
        output = -self.FCN(x.view(-1, 1, self.input_size))
        output = (1/self.system_parameters.number_of_antennas) * torch.exp(output)
        output = output.view(self.output_shape)
        return output

    def train_dataloader(self):
        train_dataset = self.InpDataset(data_path=self.data_path, normalizer=self.normalizer, mode=Mode.training, n_samples=self.n_samples, device=self.device, sqrt_laplace_matrix = self.sqrt_laplace_matrix)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader