import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from .root_model import Mode, RootDataset, CommonParameters, RootNet
from .utils import get_nu_tensor


MODEL_NAME = 'TMN'

def print_max_min(label, x):
    print(label, x.min().item(), x.max().item())



class BetaDataset(RootDataset):
    def __init__(self, data_path, normalizer, mode, n_samples, system_parameters, device):
        super(BetaDataset, self).__init__(data_path, normalizer, mode, n_samples, device)
        self.system_parameters = system_parameters

    def __getitem__(self, index):
        beta_file_name = f'betas_sample{index}.pt'
        beta_file_path = os.path.join(self.path, beta_file_name)
        beta_original = torch.load(beta_file_path)['betas'].to(dtype=torch.float32)

        nu_mat = get_nu_tensor(beta_original, self.system_parameters)
        log_nu = torch.log(torch.clamp(nu_mat, min=nu_mat.max()*1e-5))

        if self.mode == Mode.pre_processing:
            
            nu_mat = log_nu.reshape((-1,))
            return nu_mat
        
        
        log_nu_torch = log_nu.reshape((1, -1,))
        features_np = self.sc.transform(log_nu_torch)[0]
        features_torch = torch.from_numpy(features_np).to(dtype=torch.float32, device=self.device)
        features_torch = features_torch.reshape(nu_mat.shape)

        return features_torch, beta_original.to(device=self.device)

# Hyper-parameters
class HyperParameters(CommonParameters):
    # sc = StandardScaler()
    sc = MinMaxScaler()
    sc_path = os.path.join(os.getcwd(), f'{MODEL_NAME}_sc.pkl')
    training_data_path = ''

    learning_rate = 1e-3
    step_size = 150
    num_epochs = 2*step_size
    VARYING_STEP_SIZE = False

    
    InpDataSet = BetaDataset
    @classmethod
    def intialize(cls, simulation_parameters, system_parameters, is_test_mode):
        
        cls.pre_int(simulation_parameters, system_parameters, is_test_mode)

        cls.batch_size = 16
        
        if is_test_mode:
            cls.batch_size = 1  # do not delete it!
            return
        

        train_dataset = cls.InpDataSet(data_path=cls.training_data_path, normalizer=cls.sc, mode=Mode.pre_processing, n_samples=cls.n_samples, device=torch.device('cpu'), system_parameters=system_parameters)
        train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False)
        
        for log_nu in train_loader:
            with torch.no_grad():
                cls.sc.partial_fit(log_nu)

        pickle.dump(cls.sc, open(cls.sc_path, 'wb'))  # saving sc for loading later in testing phase
        print(f'{cls.sc_path} dumped!')

class Filter1(nn.Module):
    def __init__(self, num_layers, M, K):
        super(Filter1, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(1, num_layers, M, K, 1))
        self.bias = nn.Parameter(torch.Tensor(1, num_layers, M, K))
        
    def forward(self, x):
        x = x * self.weights  #element wise multiplication
        x = x.sum(dim=-1)  # addition accross i-dimension of eq. (14)
        x = x + self.bias
        return x

class Filter2(nn.Module):
    def __init__(self, num_layers, M, K):
        super(Filter2, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(1, num_layers, 1, 1))
        self.bias = nn.Parameter(torch.Tensor(1, 1, M, K))
        
    def forward(self, x):
        x = x * self.weights  #element wise multiplication
        x = x.sum(dim=1, keepdim=True)  # addition accross i-dimension of eq. (14)
        x = x + self.bias
        return x

class Filter3(nn.Module):
    def __init__(self, M, K):
        super(Filter3, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(1, M, M, 1))
        self.bias = nn.Parameter(torch.Tensor(1, M, K))
        
    def forward(self, x):
        x = x * self.weights  #element wise multiplication
        x = x.sum(dim=2)
        x = x + self.bias
        return x

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
        self.InpDataset = HyperParameters.InpDataSet

        K = HyperParameters.K
        M = HyperParameters.M

        num_layers = int(K/4)+1

        self.filter1 = Filter1(num_layers, M, K)
        self.filter2 = Filter2(num_layers, M, K)
        self.filter3 = Filter3(M, K)

        self.droupout = nn.Dropout(p=0.9)
        
        
        self.name = MODEL_NAME
        self.hardsigmoid = nn.Hardsigmoid()
        self.batch_norm2d_1 = nn.BatchNorm2d(num_layers)
        self.batch_norm2d_2 = nn.BatchNorm2d(1)
        
        self.to(self.device)
        self.batch_norm2d_1.to(self.device)
        self.batch_norm2d_2.to(self.device)


    def forward(self, x):
        # x -> b-M-K-K
        x = self.filter1(torch.unsqueeze(x, 1))# x -> b-1-M-K-K
        x = self.relu(x)
        # x = self.droupout(x)
        x = self.batch_norm2d_1(x)
        
        # x -> b-num_layers-M-K
        x = self.filter2(x)
        x = self.relu(x)
        x = self.batch_norm2d_2(x)

        # x -> b-1-M-K
        x = self.filter3(x)

        # x -> b-M-K
        output = self.hardsigmoid(x)
        return output*1e-1

    def train_dataloader(self):
        train_dataset = self.InpDataset(data_path=self.data_path, normalizer=self.normalizer, mode=Mode.training, n_samples=self.n_samples, device=self.device, system_parameters=self.system_parameters)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader