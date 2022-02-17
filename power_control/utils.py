import torch
import torch.nn as nn
import os

from utils.utils import find_the_latest_file

def compute_vmat(betas, zeta_p, T_p, phi_cross_mat, device):
    # computes Eq (5)
    # phi_cross_mat K X K
    # betas b X M X K

    den = torch.ones(betas.shape, device=device, requires_grad=False, dtype=torch.float32) + zeta_p * T_p * (betas @ (phi_cross_mat ** 2))
    v_mat = (zeta_p * T_p * (betas ** 2)) / den

    return v_mat

def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)



def load_the_latest_model_and_params_if_exists(model_folder, device, system_parameters, interm_folder, is_testing=False):
    from .model import NeuralNet
    
    model = NeuralNet(device, system_parameters, interm_folder)
    model.apply(initialize_weights)
    model_file = find_the_latest_file(model_folder)
    
    if model_file is not None:
        model.load_state_dict(torch.load(os.path.join(model_folder, model_file)))
    elif is_testing:
        from sys import exit
        print('Train the neural network before testing!')
        exit()
    print(model_file)
    return model


def individual_utility_computation(betas, mus, N, zeta_d, T_p, T_c, v_mat, phi_cross_mat, target_user):
    # Eq (16) and (17)
    # v_mat b X M X K
    # phi_cross_mat K X K
    # betas b X M X K
    # mus b X M X K

    k = target_user

    nu_mat_k = torch.einsum('k, bmk, bm -> bmk', phi_cross_mat[:, k], (torch.sqrt(v_mat) / betas), betas[:, :, k])

    nu_dot_mu = torch.einsum('bmk,bmk->bk', nu_mat_k, mus)  # B X K
    beta_k_dot_mus = torch.einsum('bm,bmk->bmk', betas[:, :, k], mus)
    b_vec = zeta_d * (nu_dot_mu)**2
    term3 = (zeta_d / N) * (torch.einsum('bmk,bmk->bk', mus, beta_k_dot_mus)) + b_vec
    b_plus_c = 1 / (N ** 2) + term3.sum(1)
    
    b = b_vec[:, k] # B
    gamma = b / (b_plus_c - b)
    SE = (1 - T_p / T_c) * torch.log(1 + gamma)  # b X 1 X 1
    return nu_mat_k, SE


def compute_smooth_min(SE_vec, tau):
    # SE_vec is of dim either K 1 or b X K
    SE_smooth_min = -(1 / tau) * torch.log((torch.exp(-tau * SE_vec)).mean(dim=-1))  # scalar or b X 1
    return SE_smooth_min

def utility_computation(betas, mus, N, geta_d, T_p, T_c, phi_cross_mat, v_mat, tau, device):
    K = betas.shape[-1]
    SE = torch.zeros((K,), device=device, requires_grad=False, dtype=torch.float32)
    for k in range(K):
        _, SE[k] = individual_utility_computation(betas, mus, N, geta_d, T_p, T_c, v_mat, phi_cross_mat, k)  # Eq (16)

    return [compute_smooth_min(SE, tau), SE]