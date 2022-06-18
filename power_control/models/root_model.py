import torch
import torch.nn as nn
import os
import datetime
from tqdm import tqdm
from enum import Enum, auto
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from sys import exit


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


class CommonParameters:
    n_samples = 1
    batch_size = 1

    learning_rate =1e-3
    gamma = 0.7
    step_size = 1
    num_epochs = 8
    eta = 1e-4
    VARYING_STEP_SIZE = True

    InpDataSet = RootDataset

    @classmethod
    def pre_int(cls, simulation_parameters, system_parameters, is_testing):
        cls.M = system_parameters.number_of_access_points
        cls.K = system_parameters.number_of_users

        
        cls.n_samples = simulation_parameters.number_of_samples
        cls.training_data_path = simulation_parameters.data_folder
        cls.validation_data_path = simulation_parameters.validation_data_folder
        cls.pre_training_data_path = '' if is_testing else simulation_parameters.pre_training_data_folder
        cls.scenario = simulation_parameters.scenario
        


class RootNet(pl.LightningModule):
    def __init__(self, device, system_parameters, interm_folder, grads):
        super(RootNet, self).__init__()
        
        self.relu = nn.ReLU()
        torch.seed()
        self.slack_variable_in = torch.rand((1,), requires_grad=True, dtype=torch.float32, device = device)
        self.slack_variable = torch.zeros((1,), requires_grad=True, dtype=torch.float32, device = device)
        # self.device = device
        self.system_parameters = system_parameters
        self.to(self.device)

        self.interm_folder = interm_folder
        self.grads = grads
        self.InpDataset = CommonParameters.InpDataSet
        self.name = None
        self.automatic_optimization = False
        
        
    def set_folder(self, model_folder):
        self.model_folder = model_folder
    
    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        beta_torch, beta_original = batch

        opt.zero_grad()
        self.slack_variable = torch.nn.functional.hardsigmoid(self.slack_variable_in)*0.1
        mus = self(beta_torch)

        with torch.no_grad():
            [mus_grads, grad_wrt_slack, utility] = self.grads(beta_original, mus, self.eta, self.slack_variable, self.device, self.system_parameters)
        
        
        self.manual_backward(mus, None, gradient=[mus_grads, grad_wrt_slack])
        opt.step()
        
        with torch.no_grad():
            temp_constraints = (1 / self.system_parameters.number_of_antennas - (torch.norm(mus, dim=2)) ** 2 - self.slack_variable ** 2)

            if torch.any(temp_constraints<0):
                print('Training constraints failed!')
                exit()
            
            loss = (-utility + (self.eta/2) * (torch.log(temp_constraints).sum(dim=-1))).mean()
        
        tensorboard_logs = {'train_loss': loss}
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        if self.VARYING_STEP_SIZE:
            sch = self.lr_schedulers()
            
            # if self.trainer.is_last_batch and self.trainer.current_epoch < (3*self.step_size+3):
            if self.trainer.is_last_batch:
                sch.step()
        

        # if (epoch_id+1) % 10 == 0:
        #     interm_model_full_path = os.path.join(self.interm_folder, f'model_{epoch_id}.pth')
        #     torch.save(self.state_dict(), interm_model_full_path)
        
        return {"loss": loss, 'log': tensorboard_logs}
    

    def validation_step(self, batch, batch_idx):
        beta_torch, beta_original = batch

        mus = self(beta_torch)

        [_, _, utility] = self.grads(beta_original, mus, self.eta, self.slack_variable, self.device, self.system_parameters) # Replace with direct utility computation
        
        
        
        temp_constraints = (1 / self.system_parameters.number_of_antennas - (torch.norm(mus, dim=2)) ** 2 - self.slack_variable ** 2)

        if torch.any(temp_constraints<0):
            print('Training constraints failed!')
            print('num_of_violations: ', (temp_constraints<0).sum())
            print('max_violations: ', ((torch.norm(mus, dim=2)) ** 2).max(), 'slack_variable: ', temp_constraints)
            exit()
        loss = (-utility + (self.eta/2) * (torch.log(temp_constraints).sum(dim=-1))).mean()

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return {"val_loss": loss}
        
    # def validation_epoch_end(self, outputs):
    #     # outputs = list of dictionaries
    #     avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     tensorboard_logs = {'avg_val_loss': avg_loss}
    #     return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def backward(self, loss, *args, **kwargs):
        mus=loss

        [mus_grads, grad_wrt_slack_batch] = kwargs['gradient']
        B = mus_grads.shape[0]
        
        mus.backward((1/B)*mus_grads)
        for grad_wrt_slack in grad_wrt_slack_batch:
            self.slack_variable.backward((1/B)*grad_wrt_slack, retain_graph=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate) 
        if self.VARYING_STEP_SIZE:
            # return [optimizer], [MultiStepLR(optimizer, milestones=[32, 64, 96, 128, 160], gamma=self.gamma)]
            return [optimizer], [StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)]
        else:
            return optimizer
    
    def train_dataloader(self):
        train_dataset = self.InpDataset(data_path=self.data_path, normalizer=self.normalizer, mode=Mode.training, n_samples=self.n_samples, device=self.device)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        return train_loader

    def val_dataloader(self):
        val_dataset = self.InpDataset(data_path=self.val_data_path, normalizer=self.normalizer, mode=Mode.training, n_samples=self.n_samples, device=self.device)
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=False)
        return val_loader


    # def training_loop(self):
    #     self.train()
    #     train_loader = self.train_dataloader()
        
        
    #     if self.VARYING_STEP_SIZE:
    #         self.opt, self.scheduler = self.configure_optimizers()
    #     else:
    #         self.opt = self.configure_optimizers()
        
    #     for epoch_id in tqdm(range(self.num_epochs)):
    #         for bacth in train_loader:
    #             utility = self.training_step(bacth, epoch_id)
            
    #         if self.VARYING_STEP_SIZE:
    #             self.scheduler.step()
                
    #         if epoch_id % 10 == 0:
    #             tqdm.write(f'\n{self.name} Utility: {-utility.min().item()}')
    
    
    def save(self):
        date_str = str(datetime.datetime.now().date()).replace(':', '_').replace('.', '_').replace('-', '_')
        time_str = str(datetime.datetime.now().time()).replace(':', '_').replace('.', '_').replace('-', '_')
        model_file_name = f'model_{date_str}_{time_str}.pth'

        model_path = os.path.join(self.model_folder, model_file_name)
        torch.save(self.state_dict(), model_path)
        
        print(model_path)
        print(f'{self.name} training Done!')
    
    # def pretrain(self):
    #     pass