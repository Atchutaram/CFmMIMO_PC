import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import pickle
from sklearn.preprocessing import StandardScaler

from .root_model import Mode, CommonParameters, RootNet


MODEL_NAME = 'FCN'

# Hyper-parameters
class HyperParameters(CommonParameters):
    sc = StandardScaler()
    sc_path = os.path.join(os.getcwd(), f'{MODEL_NAME}_sc.pkl')
    training_data_path = ''

    @classmethod
    def intialize(cls, simulation_parameters, system_parameters, is_test_mode):
        
        cls.pre_int(simulation_parameters, system_parameters, is_test_mode)
        cls.input_size = cls.M * cls.K
        cls.output_size = cls.M * cls.K
        cls.hidden_size = cls.M * cls.K
        cls.output_shape = (-1, cls.M, cls.K)
        
        if is_test_mode:
            cls.batch_size = 1
            return

        if cls.scenario == 1:
            cls.batch_size = 8 * 2
        else:
            cls.batch_size = 1
        
        

        train_dataset = cls.InpDataSet(data_path=cls.training_data_path, normalizer=cls.sc, mode=Mode.pre_processing, n_samples=cls.n_samples, device=torch.device('cpu'))
        train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False)
        
        for beta in train_loader:
            with torch.no_grad():
                cls.sc.partial_fit(beta)

        pickle.dump(cls.sc, open(cls.sc_path, 'wb'))  # saving sc for loading later in testing phase
        print(f'{cls.sc_path} dumped!')
    

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
        self.VARYING_STEP_SIZE = HyperParameters.VARYING_STEP_SIZE
        self.gamma = HyperParameters.gamma
        self.step_size = HyperParameters.step_size
        self.input_size = HyperParameters.input_size
        self.hidden_size = HyperParameters.hidden_size
        self.output_size = HyperParameters.output_size
        self.output_shape = HyperParameters.output_shape
        
        self.FCN = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size),
            nn.Hardsigmoid(),
        )
        
        self.name = MODEL_NAME
        self.to(self.device)


    def forward(self, x):
        output = self.FCN(x.view(-1, 1, self.input_size))
        output = output.view(self.output_shape)*1e-1
        return output