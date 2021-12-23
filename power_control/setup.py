import os
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle


from .model import BetaDataset, Mode


# Hyper-parameters
class CommonParameters:
    num_epochs = 200
    num_epochs = 1
    batch_size = 8
    learning_rate = 1e-3
    gamma = 0.32
    step_size = 5
    VARYING_STEP_SIZE = True
    training_data_path = ''
    sc = StandardScaler()
    sc_path = os.path.join(os.getcwd(), 'sc.pkl')
    eta = 1e-5

    def init_channel_and_system_params(self, M, K, n_samples):
        self.M  = M
        self.K  = K
        self.OUT_CH = K
        self.n_samples = n_samples
    
    def data_preprocessing(self):
        train_dataset = BetaDataset(data_path=self.training_data_path, normalizer=self.sc, mode=Mode.pre_processing, n_samples=self.n_samples)
        train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False)
        
        for beta in train_loader:
            with torch.no_grad():
                self.sc.partial_fit(beta)

        pickle.dump(self.sc, open(self.sc_path, 'wb'))  # saving sc for loading later in testing phase
        print(f'{self.sc_path} dumped!')