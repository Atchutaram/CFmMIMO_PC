import os
import torch
import torch.nn as nn
import importlib
import pickle
import math
import torch.nn.functional as F
from torch.autograd import Variable


from utils.utils import find_the_latest_file


find_import_path = lambda model_name : f"power_control.models.{model_name}_model"

def initialize_weights(m):
    # from .TMN_model import Filter1, Filter2, Filter3
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


def deploy(model, test_sample, model_name, device, **kwargs):
    import_path = find_import_path(model_name)
    module = importlib.import_module(import_path, ".")  # imports the scenarios
    
    

    with torch.no_grad():
        if model_name == 'GFT':
            test_sample = model.sqrt_laplace_matrix @ test_sample
            sc = pickle.load(open(module.HyperParameters.sc_path, 'rb'))
        else:
            test_sample = torch.log(test_sample)
            if not model_name == 'ANN' and not model_name == 'FCN':
                sc = pickle.load(open(module.HyperParameters.sc_path, 'rb'))
            

        t_shape = test_sample.shape
        test_sample = test_sample.reshape((1,-1)).to(device='cpu')
        if not model_name == 'ANN' and not model_name == 'FCN':
            test_sample = sc.transform(test_sample)[0]
            test_sample = torch.tensor(test_sample, device=device, requires_grad=False, dtype=torch.float32).view(t_shape)
        else:
            test_sample.requires_grad=False
            test_sample = test_sample.to(device=device, dtype=torch.float32).view(t_shape)
        model.eval()
        model.to(device=device)
        mus_predicted = model(test_sample)
        return mus_predicted

def initialize_hyper_params(model_name, simulation_parameters, system_parameters, is_test_mode=False):
    import_path = find_import_path(model_name)
    module = importlib.import_module(import_path, ".")  # imports the scenarios
    
    module.HyperParameters.intialize(simulation_parameters, system_parameters, is_test_mode=is_test_mode)

def load_the_latest_model_and_params_if_exists(model_name, model_folder, system_parameters, grads, device, is_testing=False):  
    import_path = find_import_path(model_name)
    module = importlib.import_module(import_path, ".")  # imports the scenarios
        
    model = module.NeuralNet(device, system_parameters, grads)
    model.apply(initialize_weights)
    model_file = find_the_latest_file(model_folder)
    
    if model_file is not None:
        model.load_state_dict(torch.load(os.path.join(model_folder, model_file)))
    elif is_testing:
        from sys import exit
        print(model_folder)
        print('Train the neural network before testing!')
        exit()
    
    if not is_testing:
        model.to(device)
        model.set_folder(model_folder)
    
    print(model_file)
    return model

def get_nu_tensor(betas, system_parameters):
    from power_control.utils import compute_vmat
    phi_cross_mat = system_parameters.phi_cross_mat
    phi_cross_mat = phi_cross_mat.to(betas.device)
    v_mat = compute_vmat(betas, system_parameters.zeta_p, system_parameters.T_p, phi_cross_mat)  # Eq (5) b X M X K
    nu_mat = torch.einsum('ik, mi, mk -> mki', phi_cross_mat, (torch.sqrt(v_mat) / betas), betas)  # Eq (14) b X M X K X K

    return nu_mat
    
def attention(query, key, value, d_k, mask=None, dropout=None):

    # query, key, value tensor are of dimension B x h x K x d_k
    # mask tensor is of dimension K x K
    
    scores = (query @ key.transpose(-2, -1)) /  math.sqrt(d_k)

    if mask is not None:
        scores = scores*mask
    
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
        
    output = scores @ value
    return output  # dimension B x h x K x d_k

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, M, dropout = 0.1):
        super().__init__()
        
        self.M = M
        self.d_k = M // heads
        if M % heads:
            from sys import exit
            print(f' The number of APs (= {M}) is not a multiple of the number of heads (={heads}). Re-adjust the number of heads.')
            exit()

        self.h = heads
        
        self.q_linear = nn.Linear(M, M)
        self.v_linear = nn.Linear(M, M)
        self.k_linear = nn.Linear(M, M)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(M, M)
    
    def forward(self, x, mask=None):
        
        # x is of dimension B x K x M
        B, _, _ = x.shape

        
        # perform linear operation and split into h heads

        query = self.q_linear(x).view(B, -1, self.h, self.d_k)
        key = self.k_linear(x).view(B, -1, self.h, self.d_k)
        value = self.v_linear(x).view(B, -1, self.h, self.d_k)
        
        # transpose to get dimensions B x h x K x d_k
       
        query = query.transpose(1,2)
        key = key.transpose(1,2)
        value = value.transpose(1,2)

        scores = attention(query, key, value, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(B, -1, self.M)

        output = self.out(concat)
        return output  # dimensions B x K x M


class FeedForward(nn.Module):

    def __init__(self, M, dropout = 0.1):
        super().__init__() 
        d_mid = int(4 * M / (1-dropout))

        self.linear_1 = nn.Linear(M, d_mid)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_mid, M)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class Norm(nn.Module):
    
    def __init__(self, size, eps = 1e-6):
        super().__init__()
    
        self.size = size
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=[-2, -1], keepdim=True)) / (x.std(dim=[-2, -1], keepdim=True) + self.eps) + self.bias
        return norm

class EncoderLayer(nn.Module):
    
    def __init__(self, M, heads, dropout = 0.1):
        super().__init__()
        self.norm_1 = Norm(M)
        self.norm_2 = Norm(M)
        self.attn = MultiHeadAttention(heads, M, dropout)
        self.ff = FeedForward(M, dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x

class PositionalEncoder(nn.Module):
    def __init__(self, M, max_seq_len = 80):
        super().__init__()
        self.M = M
        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, M)
        for pos in range(max_seq_len):
            for i in range(0, M, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/M)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/M)))
                
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.M)
        #add constant to embedding
        seq_len = x.shape[1]
        x = x + Variable(self.pe[:,:seq_len], requires_grad=False).to(device=x.device)
        return x