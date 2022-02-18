import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import pickle
from sklearn.preprocessing import StandardScaler

from .root_model import Mode, RootDataset, CommonParameters, RootNet


MODEL_NAME = 'CNN'

class BetaDataset(RootDataset):
    def __init__(self, data_path, normalizer, mode, n_samples, device):
        super(BetaDataset, self).__init__(data_path, normalizer, mode, n_samples, device)

    def __getitem__(self, index):
        beta_file_name = f'betas_sample{index}.pt'
        beta_file_path = os.path.join(self.path, beta_file_name)
        beta_original = torch.load(beta_file_path)['betas'].to(dtype=torch.float32)
        if self.mode == Mode.pre_processing:
            beta = torch.log(beta_original.reshape((-1,)))
            return beta
        
        beta_torch = torch.log(beta_original)
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
    def data_preprocessing(cls, M, K, n_samples, training_data_path, scenario):
        cls.M  = M
        cls.K  = K
        cls.input_size = K * M
        if scenario == 1:
            cls.batch_size = 8 * 2
            cls.OUT_CH = 600
        else:
            cls.batch_size = 1
            cls.OUT_CH = 4
        
        cls.n_samples = n_samples
        cls.training_data_path = training_data_path

        train_dataset = BetaDataset(data_path=cls.training_data_path, normalizer=cls.sc, mode=Mode.pre_processing, n_samples=cls.n_samples, device=torch.device('cpu'))
        train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False)
        
        for beta in train_loader:
            with torch.no_grad():
                cls.sc.partial_fit(beta)

        pickle.dump(cls.sc, open(cls.sc_path, 'wb'))  # saving sc for loading later in testing phase
        print(f'{cls.sc_path} dumped!')
    
    @classmethod
    def test_setup(cls, M, K, scenario):
        cls.M  = M
        cls.K  = K
        cls.input_size = K * M
        
        if scenario == 1:
            cls.OUT_CH = 600
        else:
            cls.OUT_CH = 4

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

        K = HyperParameters.K
        M = HyperParameters.M
        OUT_CH = HyperParameters.OUT_CH
        self.OUT_CH = OUT_CH
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, round(OUT_CH / 4), (3, 3), stride=(2, 2), padding=(1, 1)), # input channels, output channels, kernel size
            nn.ReLU(),
            nn.Conv2d(round(OUT_CH / 4), round(OUT_CH / 2), (3, 3), stride=(2, 2), padding=(1, 1)), # input channels, output channels, kernel size
            nn.ReLU(),
            nn.Conv2d(round(OUT_CH / 2), OUT_CH, (round(M / 4), round(K / 4))),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(OUT_CH, round(OUT_CH / 2), (round(M / 4), round(K / 4))),
            nn.ReLU(),
            nn.ConvTranspose2d(round(OUT_CH / 2), round(OUT_CH / 4), (3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(round(OUT_CH / 4), 1, (3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
        )
        self.linear1 = nn.Linear(OUT_CH, OUT_CH)
        
        self.name = MODEL_NAME
        self.to(self.device)


    def forward(self, x):
        encoded = self.encoder(torch.unsqueeze(x, 1))
        encoded_shape = encoded.shape
        encoded = self.relu(self.linear1(encoded.view(-1, 1, self.OUT_CH)))
        decoded = self.decoder(encoded.view(encoded_shape))
        decoded = -self.relu(decoded)  # so max final output after torch.exp is always between 0 and 1. This conditioning helps regularization.

        out = (1/self.system_parameters.number_of_antennas) * torch.exp(decoded)
        out = torch.squeeze(out, dim=1)
        return out

    def train_dataloader(self):
        train_dataset = BetaDataset(data_path=self.data_path, normalizer=self.normalizer, mode=Mode.training, n_samples=self.n_samples, device=self.device)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader