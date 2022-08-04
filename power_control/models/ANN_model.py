import torch
import os
from sklearn.preprocessing import StandardScaler


from .root_model import RootDataset, CommonParameters, RootNet
from .utils import EncoderLayer, Norm
from utils.utils import tensor_max_min_print


MODEL_NAME = 'ANN'

class BetaDataset(RootDataset):
    def __init__(self, data_path, phi_orth, normalizer, mode, n_samples):
        self.path = data_path
        _, _, files = next(os.walk(self.path))
        self.n_samples = min(len(list(filter(lambda k: 'betas' in k, files))), n_samples)
        self.sc = normalizer
        self.mode = mode
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

# Hyper-parameters
class HyperParameters(CommonParameters):
    sc = StandardScaler()
    sc_path = os.path.join(os.getcwd(), f'{MODEL_NAME}_sc.pkl')
    training_data_path = ''
    InpDataSet = BetaDataset

    @classmethod
    def intialize(cls, simulation_parameters, system_parameters, is_test_mode):
        
        cls.pre_int(simulation_parameters, system_parameters)

        number_of_micro_batches = torch.cuda.device_count()  # This is to handle data parallelism
        if not number_of_micro_batches:
            number_of_micro_batches = 1

        if cls.scenario == 0:
            batch_size = 8 * 2
            cls.batch_size = int(batch_size/number_of_micro_batches)
        elif cls.scenario == 1:
            batch_size = 8 * 2
            cls.batch_size = int(batch_size/number_of_micro_batches)
        else:
            cls.batch_size = 1
        
        if is_test_mode:
            cls.batch_size = 1
            return
    
    

class NeuralNet(RootNet):
    def __init__(self, system_parameters, grads):
        super(NeuralNet, self).__init__(system_parameters, grads)
        
        self.n_samples = HyperParameters.n_samples
        self.num_epochs = HyperParameters.num_epochs
        self.eta = HyperParameters.eta
        self.data_path = HyperParameters.training_data_path
        self.val_data_path = HyperParameters.validation_data_path
        self.normalizer = HyperParameters.sc
        self.batch_size = HyperParameters.batch_size
        self.learning_rate = HyperParameters.learning_rate
        self.VARYING_STEP_SIZE = HyperParameters.VARYING_STEP_SIZE
        self.gamma = HyperParameters.gamma
        self.step_size = HyperParameters.step_size
        self.InpDataset = HyperParameters.InpDataSet

        K = HyperParameters.K
        M = HyperParameters.M
        
        
        dropout = 0.5
        heads = 5

        self.layer1 = EncoderLayer(M, heads=heads, dropout=dropout)
        self.layer2 = EncoderLayer(M, heads=heads, dropout=dropout)
        self.layer3 = EncoderLayer(M, heads=heads, dropout=dropout)
        self.layer4 = EncoderLayer(M, heads=heads, dropout=dropout)
        self.layer5 = EncoderLayer(M, heads=heads, dropout=dropout)
        self.layer6 = EncoderLayer(M, heads=heads, dropout=dropout)


        self.name = MODEL_NAME


    def forward(self, x):
        x, phi_cross_mat = x
        mask = torch.unsqueeze(phi_cross_mat**2, dim=1)
        x = x.transpose(1,2).contiguous()
        
        x = self.layer1(x, mask=mask)
        
        x = self.layer2(x, mask=mask)
        
        x = self.layer3(x, mask=mask)
        
        x = self.layer4(x, mask=mask)
        
        x = self.layer5(x, mask=mask)
        
        x = self.layer6(x, mask=mask)
        x = torch.nn.functional.hardsigmoid(x)

        return x.transpose(1,2).contiguous()*torch.nn.functional.hardsigmoid(self.multiplication_factor_in).view(1,-1,1)*self.N_inv_root