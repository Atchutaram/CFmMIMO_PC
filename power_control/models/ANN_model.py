import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import pickle
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import StepLR
import math


from .root_model import Mode, RootDataset, CommonParameters, RootNet
from .utils import EncoderLayer, Norm,  PositionalEncoder


MODEL_NAME = 'ANN'

# class BetaDataset(RootDataset):
#     def __init__(self, data_path, normalizer, mode, n_samples, device):
#         super(BetaDataset, self).__init__(data_path, normalizer, mode, n_samples, device)

#     def __getitem__(self, index):
#         beta_file_name = f'betas_sample{index}.pt'
#         beta_file_path = os.path.join(self.path, beta_file_name)
#         beta_original = torch.load(beta_file_path)['betas'].to(dtype=torch.float32)

#         b_max = beta_original.max()
#         beta_torch = 20*(beta_original/b_max).to(dtype=torch.float32, device=self.device)
#         return beta_torch, beta_original.to(device=self.device)
class BetaDataset(RootDataset):
    def __init__(self, data_path, normalizer, mode, n_samples, device):
        self.path = data_path
        _, _, files = next(os.walk(self.path))
        self.n_samples = min(len(list(filter(lambda k: 'betas' in k, files))), n_samples)
        self.sc = normalizer
        self.mode = mode
        self.device = device
        
    def __getitem__(self, index):
        beta_file_name = f'betas_sample{index}.pt'
        beta_file_path = os.path.join(self.path, beta_file_name)
        beta_original = torch.load(beta_file_path)['betas'].to(dtype=torch.float32)
        
        beta_torch = torch.log(beta_original)
        # beta_torch = beta_original
        beta_torch = beta_torch.to(dtype=torch.float32, device=self.device)
        beta_torch = beta_torch.reshape(beta_original.shape)

        return beta_torch, beta_original.to(device=self.device)

    def __len__(self):
        return self.n_samples

# Hyper-parameters
class HyperParameters(CommonParameters):
    sc = StandardScaler()
    sc_path = os.path.join(os.getcwd(), f'{MODEL_NAME}_sc.pkl')
    training_data_path = ''
    InpDataSet = BetaDataset

    # learning_rate = 1e-6
    # eta = 1e-6
    # step_size = 800
    # num_epochs = 2*step_size
    # VARYING_STEP_SIZE = True

    @classmethod
    def intialize(cls, simulation_parameters, system_parameters, is_test_mode):
        
        cls.pre_int(simulation_parameters, system_parameters, is_test_mode)

        if cls.scenario == 0:
            cls.batch_size = 8 * 2
        elif cls.scenario == 1:
            cls.batch_size = 8 * 2
        else:
            cls.batch_size = 1
        
        if is_test_mode:
            cls.batch_size = 1
            return
        

        # train_dataset = cls.InpDataSet(data_path=cls.training_data_path, normalizer=cls.sc, mode=Mode.pre_processing, n_samples=cls.n_samples, device=torch.device('cpu'))
        # train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False)
        
        # for beta in train_loader:
        #     with torch.no_grad():
        #         cls.sc.partial_fit(beta)

        # pickle.dump(cls.sc, open(cls.sc_path, 'wb'))  # saving sc for loading later in testing phase
        # print(f'{cls.sc_path} dumped!')
    
    

class NeuralNet(RootNet):
    def __init__(self, device, system_parameters, interm_folder, grads):
        super(NeuralNet, self).__init__(device, system_parameters, interm_folder, grads)
        
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
        self.InpDataset = HyperParameters.InpDataSet

        K = HyperParameters.K
        M = HyperParameters.M
        
        
        dropout = 0.2
        heads = 5

        # self.pe = PositionalEncoder(M, max_seq_len = K)
        self.layer1 = EncoderLayer(M, heads=heads, dropout=dropout)
        self.layer2 = EncoderLayer(M, heads=heads, dropout=dropout)
        self.layer3 = EncoderLayer(M, heads=heads, dropout=dropout)
        self.layer4 = EncoderLayer(M, heads=heads, dropout=dropout)
        self.layer5 = EncoderLayer(M, heads=heads, dropout=dropout)
        self.layer6 = EncoderLayer(M, heads=heads, dropout=dropout)
        self.norm = Norm(M)
        
        self.name = MODEL_NAME
        self.to(self.device)


    def forward(self, x):
        x = x.transpose(1,2).contiguous()
        # x = self.pe(x)
        
        x = self.layer1(x, mask=self.system_parameters.phi_cross_mat**2)
        
        x = self.layer2(x, mask=self.system_parameters.phi_cross_mat**2)
        
        x = self.layer3(x, mask=self.system_parameters.phi_cross_mat**2)
        
        # x = self.layer4(x, mask=self.system_parameters.phi_cross_mat**2)
        
        # x = self.layer5(x, mask=self.system_parameters.phi_cross_mat**2)
        
        # x = self.layer6(x, mask=self.system_parameters.phi_cross_mat**2)

        x = self.norm(x)
        x = torch.sigmoid(x)

        return x.transpose(1,2).contiguous()*1e-1
    
    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    #     if self.VARYING_STEP_SIZE:
    #         return optimizer, StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
    #     else:
    #         return optimizer