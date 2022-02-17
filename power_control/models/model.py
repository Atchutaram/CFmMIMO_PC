import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import pytorch_lightning as pl

from .dataset_handling import BetaDataset, Mode
from .nn_setup import CommonParameters


class NeuralNet(pl.LightningModule):
    def __init__(self, system_parameters, interm_folder, grads):
        super(NeuralNet, self).__init__()
        K = CommonParameters.K
        M = CommonParameters.M
        OUT_CH = CommonParameters.OUT_CH
        
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
        self.automatic_optimization = False

        self.slack_variable_in = torch.rand((1,), requires_grad=True, dtype=torch.float32)
        self.slack_variable = torch.zeros((1,), requires_grad=True, dtype=torch.float32)
        self.n_samples = CommonParameters.n_samples
        self.system_parameters = system_parameters

        self.interm_folder = interm_folder
        self.grads = grads
        self.epoch_idx = 0

    def forward(self, x):
        encoded = self.encoder(torch.unsqueeze(x, 1))
        decoded = self.decoder(encoded)
        decoded = -self.relu(decoded)  # so max final output after torch.exp is always between 0 and 1. This conditioning helps regularization.

        out = (1/self.system_parameters.number_of_antennas) * torch.exp(decoded)
        out = torch.squeeze(out, dim=1)
        return out
    
    def training_step(self, batch, batch_idx):        
        if batch_idx == 0 and self.epoch_idx == 0:
            self.slack_variable_in = self.slack_variable_in.to(self.device)
        
        self.slack_variable = torch.tanh(self.slack_variable_in)

        
        opt = self.optimizers(use_pl_optimizer=True)
        opt.zero_grad()

        betas, beta_original = batch
        mus = self(betas)

        with torch.no_grad():
            [mus_grads, grad_wrt_slack, utility] = self.grads(beta_original, mus, CommonParameters.eta, self.slack_variable, self.device, self.system_parameters)
            self.log('train_loss', -utility.mean(), on_step=True, on_epoch=True, prog_bar=True)
            self.epoch_idx += (batch_idx == 0)

            if self.epoch_idx % 10 == 0 and self.epoch_idx > 0:
                interm_model_full_path = os.path.join(self.interm_folder, f'model_{self.epoch_idx}.pth')
                torch.save(self.state_dict(), interm_model_full_path)

        self.manual_backward(mus, None, gradient=[mus_grads, grad_wrt_slack])
        opt.step()

    def backward(self, loss, optimizer, optimizer_idx, *args, **kwargs):
        [mus_grads, grad_wrt_slack_batch] = kwargs["gradient"]
        
        loss.backward(mus_grads)
        for grad_wrt_slack in grad_wrt_slack_batch:
            self.slack_variable.backward(grad_wrt_slack, retain_graph=True)

    def train_dataloader(self):
        train_dataset = BetaDataset(data_path=CommonParameters.training_data_path, normalizer=CommonParameters.sc, mode=Mode.training, n_samples=self.n_samples, device=self.device)
        train_loader = DataLoader(dataset=train_dataset, batch_size=CommonParameters.batch_size, shuffle=False)
        return train_loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=CommonParameters.learning_rate)
        if CommonParameters.VARYING_STEP_SIZE:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=CommonParameters.step_size, gamma=CommonParameters.gamma)
            return [optimizer], [scheduler]
        else:
            return optimizer