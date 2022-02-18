import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os
from enum import Enum, auto
import pickle
from sklearn.preprocessing import StandardScaler
import datetime
from tqdm import tqdm


MODEL_NAME = 'CNN'

class Mode(Enum):
    pre_processing = auto()
    training = auto()


class BetaDataset(Dataset):
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
        if self.mode == Mode.pre_processing:
            beta = torch.log(beta_original.reshape((-1,)))
            return beta
        
        beta_torch = torch.log(beta_original)
        beta_torch = beta_torch.reshape((1, -1,))
        beta_torch = self.sc.transform(beta_torch)[0]
        beta_torch = torch.from_numpy(beta_torch).to(dtype=torch.float32, device=self.device)
        beta_torch = beta_torch.reshape(beta_original.shape)

        return beta_torch, beta_original.to(device=self.device)

    def __len__(self):
        return self.n_samples


# Hyper-parameters
class HyperParameters:
    n_samples = 1
    batch_size = 1
    num_epochs = 200
    learning_rate = 1e-4

    gamma = 0.32
    step_size = 5
    sc = StandardScaler()
    sc_path = os.path.join(os.getcwd(), f'{MODEL_NAME}_sc.pkl')
    eta = 1e-5

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

class NeuralNet(nn.Module):
    def __init__(self, device, system_parameters, interm_folder, grads):
        super(NeuralNet, self).__init__()
        K = HyperParameters.K
        M = HyperParameters.M
        OUT_CH = HyperParameters.OUT_CH
        
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
        

        self.relu = nn.ReLU()

        self.slack_variable_in = torch.rand((1,), requires_grad=True, dtype=torch.float32, device = device)
        self.slack_variable = torch.zeros((1,), requires_grad=True, dtype=torch.float32, device = device)
        self.device = device
        self.n_samples = HyperParameters.n_samples
        self.system_parameters = system_parameters
        self.to(self.device)

        self.interm_folder = interm_folder
        self.grads = grads
        self.opt = None
        
        self.num_epochs = HyperParameters.num_epochs
        self.name = MODEL_NAME
    
    def set_folder(self, model_folder):
        self.model_folder = model_folder
    
    def forward(self, x):
        encoded = self.encoder(torch.unsqueeze(x, 1))
        decoded = self.decoder(encoded)
        decoded = -self.relu(decoded)  # so max final output after torch.exp is always between 0 and 1. This conditioning helps regularization.

        out = (1/self.system_parameters.number_of_antennas) * torch.exp(decoded)
        out = torch.squeeze(out, dim=1)
        return out
    
    def training_step(self, batch, epoch_id):
        beta_torch, beta_original = batch

        self.opt.zero_grad()
        self.slack_variable = torch.tanh(self.slack_variable_in)
        mus = self(beta_torch)

        with torch.no_grad():
            [mus_grads, grad_wrt_slack, utility] = self.grads(beta_original, mus, HyperParameters.eta, self.slack_variable, self.device, self.system_parameters)
        
        self.backward(mus, gradient=[mus_grads, grad_wrt_slack])
        self.opt.step()

        if epoch_id % 10 == 0 and epoch_id > 0:
            interm_model_full_path = os.path.join(self.interm_folder, f'model_{epoch_id}.pth')
            torch.save(self.state_dict(), interm_model_full_path)
        
        return utility

    def backward(self, mus, gradient):

        [mus_grads, grad_wrt_slack_batch] = gradient
        
        mus.backward(mus_grads)
        for grad_wrt_slack in grad_wrt_slack_batch:
            self.slack_variable.backward(grad_wrt_slack, retain_graph=True)

    def train_dataloader(self):
        train_dataset = BetaDataset(data_path=HyperParameters.training_data_path, normalizer=HyperParameters.sc, mode=Mode.training, n_samples=self.n_samples, device=self.device)
        train_loader = DataLoader(dataset=train_dataset, batch_size=HyperParameters.batch_size, shuffle=False)
        return train_loader

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=HyperParameters.learning_rate)
    
    def train(self):
        train_loader = self.train_dataloader()
        
        self.opt = self.configure_optimizers()
        
        for epoch_id in tqdm(range(self.num_epochs)):
            for bacth in train_loader:
                utility = self.training_step(bacth, epoch_id)
                
            if epoch_id % 10 == 0:
                tqdm.write(f'\nUtility: {-utility.mean().item()}')
        
        date_str = str(datetime.datetime.now().date()).replace(':', '_').replace('.', '_').replace('-', '_')
        time_str = str(datetime.datetime.now().time()).replace(':', '_').replace('.', '_').replace('-', '_')
        model_file_name = f'model_{date_str}_{time_str}.pth'

        model_path = os.path.join(self.model_folder, model_file_name)
        torch.save(self.state_dict(), model_path)
        
        print(model_path)
        print(f'{self.name} training Done!')