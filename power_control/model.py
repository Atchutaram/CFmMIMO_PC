import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

from .dataset_handling import BetaDataset, Mode
from .setup import CommonParameters
from channel_env.gradient_handler import grads

class NeuralNet(nn.modules):
    def __init__(self, device):
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
        self.slack_variable_in = torch.rand((1,), requires_grad=True, dtype=torch.float32, device = device)
        self.slack_variable = torch.zeros((1,), requires_grad=True, dtype=torch.float32, device = device)
        self.device = device

    def forward(self, x):
        encoded = self.encoder(torch.unsqueeze(x, 1))
        decoded = self.decoder(encoded)
        decoded = -self.relu(decoded)  # so max final output after torch.exp is always between 0 and 1. 
        # Ideally, this should be between 0 and 1/N. We consider N=1 in our example anyway. This conditioning helps regularization.

        out = torch.exp(decoded)
        out = torch.squeeze(out, dim=1)
        return out
    
    def training_step(self, beta_torch, beta_original, opt, epoch_id):
        self.slack_variable = torch.tanh(self.slack_variable_in)
        mus = self(beta_torch)

        [y_grads, grad_wrt_slack, utility] = grads(beta_original, mus, CommonParameters.eta, self.slack_variable, self.device)
        
        opt.zero_grad()
        self.backward(beta_torch, gradient=[y_grads, grad_wrt_slack])
        opt.step()

        if epoch_id % 10 == 0 and epoch_id > 0:
            interm_model_full_path = os.path.join(os.getcwd(), 'interm_models', f'model_{epoch_id}.pth')
            torch.save(self.state_dict(), interm_model_full_path)
        
        return utility

    def backward(self, y, gradient):

        [y_grads, grad_wrt_slack_batch] = gradient
        y.backward(y_grads)
        for grad_wrt_slack in grad_wrt_slack_batch:
            self.slack_variable.backward(grad_wrt_slack, retain_graph=True)

    def train_dataloader(self):
        train_dataset = BetaDataset(data_path=CommonParameters.training_data_path, normalizer=CommonParameters.sc, mode=Mode.training, n_samples=self.n_samples)
        # Data loader
        train_loader = DataLoader(dataset=train_dataset, batch_size=CommonParameters.batch_size, shuffle=False)
        return train_loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=CommonParameters.learning_rate)
        if CommonParameters.VARYING_STEP_SIZE:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=CommonParameters.step_size, gamma=CommonParameters.gamma)
            return [optimizer], [scheduler]
        else:
            return optimizer