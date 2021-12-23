import torch
from torch.utils.data import Dataset
import os
from enum import Enum, auto

class Mode(Enum):
    pre_processing = auto()
    training = auto()

class BetaDataset(Dataset):
    def __init__(self, data_path, normalizer, mode, n_samples):
        self.path = data_path
        _, _, files = next(os.walk(self.path))
        self.n_samples = min(len(list(filter(lambda k: 'betas' in k, files))), n_samples)
        self.sc = normalizer
        self.mode = mode

    def __getitem__(self, index):
        beta_file_name = f'betas_sample{index}.pt'
        beta_file_path = os.path.join(self.path, beta_file_name)
        beta_original = torch.load(beta_file_path)['betas'].to(dtype=torch.float32)
        if self.mode == Mode.pre_processing:
            beta = torch.log(beta_original.reshape((-1,)))
            return beta
        
        beta_torch = torch.log(beta_original)
        beta_torch = torch.unsqueeze(beta_torch.reshape((-1,)), 0)
        beta_torch = self.sc.transform(beta_torch)[0]
        beta_torch = torch.from_numpy(beta_torch).to(dtype=torch.float32)
        beta_torch = beta_torch.reshape(beta_original.shape)

        return beta_torch, beta_original

    def __len__(self):
        return self.n_samples
