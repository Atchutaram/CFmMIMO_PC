import os
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle


from .dataset_handling import BetaDataset, Mode


# Hyper-parameters
class CommonParameters:
    n_samples = 1
    batch_size = 1
    num_epochs = 200
    learning_rate = 1e-4

    gamma = 0.32
    step_size = 5
    VARYING_STEP_SIZE = False
    sc = StandardScaler()
    sc_path = os.path.join(os.getcwd(), 'sc.pkl')
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
        