import torch
import torch.nn as nn
import os
import datetime
from tqdm import tqdm
from enum import Enum, auto
from torch.utils.data import Dataset


class Mode(Enum):
    pre_processing = auto()
    training = auto()


class RootDataset(Dataset):
    def __init__(self, data_path, normalizer, mode, n_samples, device):
        self.path = data_path
        _, _, files = next(os.walk(self.path))
        self.n_samples = min(len(list(filter(lambda k: 'betas' in k, files))), n_samples)
        self.sc = normalizer
        self.mode = mode
        self.device = device

    def __len__(self):
        return self.n_samples


class CommonParameters:
    n_samples = 1
    batch_size = 1
    num_epochs = 200
    learning_rate = 1e-4

    gamma = 0.32
    step_size = 5
    eta = 1e-5

class RootNet(nn.Module):
    def __init__(self, device, system_parameters, interm_folder, grads):
        super(RootNet, self).__init__()
        
        self.relu = nn.ReLU()

        self.slack_variable_in = torch.rand((1,), requires_grad=True, dtype=torch.float32, device = device)
        self.slack_variable = torch.zeros((1,), requires_grad=True, dtype=torch.float32, device = device)
        self.device = device
        self.system_parameters = system_parameters
        self.to(self.device)

        self.interm_folder = interm_folder
        self.grads = grads
        
    
    def set_folder(self, model_folder):
        self.model_folder = model_folder
    
    def training_step(self, batch, epoch_id):
        beta_torch, beta_original = batch

        self.opt.zero_grad()
        self.slack_variable = torch.tanh(self.slack_variable_in)
        mus = self(beta_torch)

        with torch.no_grad():
            [mus_grads, grad_wrt_slack, utility] = self.grads(beta_original, mus, self.eta, self.slack_variable, self.device, self.system_parameters)
        
        self.backward(mus, gradient=[mus_grads, grad_wrt_slack])
        self.opt.step()

        if (epoch_id+1) % 10 == 0:
            interm_model_full_path = os.path.join(self.interm_folder, f'model_{epoch_id}.pth')
            torch.save(self.state_dict(), interm_model_full_path)
        
        return utility

    def backward(self, mus, gradient):

        [mus_grads, grad_wrt_slack_batch] = gradient
        
        mus.backward(mus_grads)
        for grad_wrt_slack in grad_wrt_slack_batch:
            self.slack_variable.backward(grad_wrt_slack, retain_graph=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
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