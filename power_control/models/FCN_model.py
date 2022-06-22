import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import pickle
from sklearn.preprocessing import StandardScaler

from .root_model import Mode, RootDataset, CommonParameters, RootNet
from .utils import Norm


MODEL_NAME = 'FCN'

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

    @classmethod
    def intialize(cls, simulation_parameters, system_parameters, is_test_mode):
        
        cls.pre_int(simulation_parameters, system_parameters, is_test_mode)
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
            # cls.learning_rate = 1e-5
            cls.batch_size = 8 * 2
        elif cls.scenario == 1:
            cls.batch_size = 8 * 2
        else:
            cls.batch_size = 1
        
        

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
        self.input_size = HyperParameters.input_size
        self.hidden_size = HyperParameters.hidden_size
        self.output_size = HyperParameters.output_size
        self.output_shape = HyperParameters.output_shape
        self.InpDataset = HyperParameters.InpDataSet
        self.dropout = HyperParameters.dropout
        
        # self.FCN = nn.Sequential(
        #     Norm(self.input_size),
        #     nn.Linear(self.input_size, self.hidden_size),
        #     nn.ReLU(),
        #     nn.Dropout(p=self.dropout),

        #     # Norm(self.hidden_size),
        #     # nn.Linear(self.hidden_size, self.hidden_size),
        #     # nn.ReLU(),
        #     # nn.Dropout(p=self.dropout),

        #     Norm(self.hidden_size),
        #     nn.Linear(self.hidden_size, self.output_size),
        #     nn.Sigmoid(),
        # )
        
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
        self.to(self.device)


    def forward(self, x):
        output = self.hidden_layer_1(x.view(-1, 1, self.input_size))
        output = self.hidden_layer_2(output)
        output = self.output_layer(output)
        output = output.view(self.output_shape)*1e-1
        return output