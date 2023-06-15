import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import math

class RootDataset(Dataset):
    def __init__(self, data_path, phi_orth, n_samples):
        self.path = data_path
        _, _, files = next(os.walk(self.path))
        self.n_samples = min(len(list(filter(lambda k: 'betas' in k, files))), n_samples)
        self.phi_orth = phi_orth
        
    def __getitem__(self, index):
        beta_file_name = f'betas_sample{index}.pt'
        beta_file_path = os.path.join(self.path, beta_file_name)
        m = torch.load(beta_file_path)
        beta_original = m['betas'].to(dtype=torch.float32)
        pilot_sequence = m['pilot_sequence'].to(dtype=torch.int32)

        phi = torch.index_select(self.phi_orth, 0, pilot_sequence)
        phi_cross_mat = torch.abs(phi.conj() @ phi.T)
        
        beta_torch = torch.log(beta_original)
        beta_torch = beta_torch.to(dtype=torch.float32)
        beta_torch = beta_torch.reshape(beta_original.shape)

        return phi_cross_mat, beta_torch, beta_original

    def __len__(self):
        return self.n_samples


class CommonParameters:
    n_samples = 1
    batch_size = 16
    num_epochs = 4*4

    learning_rate =3e-4
    
    # Params related to varying step size
    VARYING_STEP_SIZE = True
    gamma = 0.75  # LR decay multiplication factor
    step_size = 1 # for varying lr
    training_data_path = ''

    @classmethod
    def pre_int(cls, simulation_parameters, system_parameters):
        cls.M = system_parameters.number_of_access_points
        cls.K = system_parameters.number_of_users
        
        cls.n_samples = simulation_parameters.number_of_samples
        cls.training_data_path = simulation_parameters.data_folder
        cls.validation_data_path = simulation_parameters.validation_data_folder
        cls.scenario = simulation_parameters.scenario
        cls.dropout = 0
        
        if (simulation_parameters.operation_mode == 2) or (cls.scenario > 2):
            cls.batch_size = 1  # for either large-scale systems or for testing mode
        


class RootNet(pl.LightningModule):
    def __init__(self, system_parameters, grads):
        super(RootNet, self).__init__()
        self.save_hyperparameters()

        torch.seed()
        self.automatic_optimization = False
        
        self.InpDataset = RootDataset
        self.N = system_parameters.number_of_antennas
        self.N_inv_root = 1/math.sqrt(self.N)
        self.system_parameters = system_parameters
        self.grads = grads
        self.relu = nn.ReLU()
        self.name = None
        
    
    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        phi_cross_mat, beta_torch, beta_original = batch

        opt.zero_grad()
        mus = self([beta_torch, phi_cross_mat])

        with torch.no_grad():
            [mus_grads, utility] = self.grads(beta_original, mus, self.device, self.system_parameters, phi_cross_mat)
        
        self.manual_backward(mus, None, gradient=mus_grads)
        opt.step()
        
        with torch.no_grad():
            loss = -utility.mean() # loss is negative of the utility
        tensorboard_logs = {'train_loss': loss}
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss, 'log': tensorboard_logs}
    

    def validation_step(self, batch, batch_idx):
        phi_cross_mat, beta_torch, beta_original = batch

        mus = self([beta_torch, phi_cross_mat])

        [_, utility] = self.grads(beta_original, mus, self.device, self.system_parameters, phi_cross_mat)
        loss = -utility.mean()
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return {"val_loss": loss}

    def on_train_epoch_end(self, *args, **kwargs):
        # We are using this method to manipulate the learning rate according to our needs
        skip_length = 1
        LL=7
        UL=10
        if self.VARYING_STEP_SIZE:
            sch = self.lr_schedulers()
            if LL<=self.current_epoch<=UL and (self.current_epoch % skip_length==0):
                sch.step()
            # print('\nEpoch end', sch.get_last_lr())
        else:
            # print('\nEpoch end', self.learning_rate)
            pass

        

    def backward(self, loss, *args, **kwargs):
        mus=loss
        mus_grads= kwargs['gradient']
        B = mus_grads.shape[0]
        mus.backward((1/B)*mus_grads)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate) 
        if self.VARYING_STEP_SIZE:
            return [optimizer], [StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)]
        else:
            return optimizer
    
    def train_dataloader(self):
        train_dataset = self.InpDataset(data_path=self.data_path, phi_orth=self.system_parameters.phi_orth, n_samples=self.n_samples)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        return train_loader

    def val_dataloader(self):
        val_dataset = self.InpDataset(data_path=self.val_data_path, phi_orth=self.system_parameters.phi_orth, n_samples=self.n_samples)
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=False)
        return val_loader