import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import pickle
from sklearn.preprocessing import StandardScaler

from .root_model import Mode, RootDataset, CommonParameters, RootNet


MODEL_NAME = 'TDN'


# Hyper-parameters
class HyperParameters(CommonParameters):
    sc = StandardScaler()
    sc_path = os.path.join(os.getcwd(), f'{MODEL_NAME}_sc.pkl')
    training_data_path = ''

    @classmethod
    def intialize(cls, simulation_parameters, system_parameters, is_test_mode):
        cls.M = system_parameters.number_of_access_points
        cls.K = system_parameters.number_of_users
        cls.hid = 50*cls.K

        
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
        
        train_dataset = BetaDataset(data_path=cls.training_data_path, normalizer=cls.sc, mode=Mode.pre_processing, n_samples=cls.n_samples, device=torch.device('cpu'))
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
        K = HyperParameters.K
        M = HyperParameters.M
        hid = HyperParameters.hid
        
        self.FCNs = nn.ModuleList()
        for _ in range(M):
            self.FCNs.append(
                nn.Sequential(
                    nn.Linear(K, hid),
                    nn.ReLU(),
                    nn.Linear(hid, hid),
                    nn.ReLU(),
                    nn.Linear(hid, K),
                    )
            )

        self.name = MODEL_NAME
        self.to(self.device)


    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        decoded = []
        for m, FCN in enumerate(self.FCNs):
            decoded_temp = FCN(x[:, 0, m, :])
            decoded_temp = -self.relu(decoded_temp)  # so max final output after torch.exp is always between 0 and 1. This conditioning helps regularization.
            decoded_temp = (1/self.system_parameters.number_of_antennas) * torch.exp(decoded_temp)
            decoded.append(torch.unsqueeze(decoded_temp, 0))
        
        decoded = torch.transpose(torch.cat(decoded), 0, 1).to(device=self.device)
        return decoded