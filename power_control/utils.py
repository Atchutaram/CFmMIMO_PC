import torch
import torch.nn as nn
import os

from .model import NeuralNet


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


def find_the_latest_file(model_folder):
    import glob
    file = None
    list_of_files = glob.glob(os.path.join(model_folder, '*'))
    if list_of_files:
        file = max(list_of_files, key=os.path.getctime)
    return file

def load_the_latest_model_and_params_if_exists(model_folder, device, is_testing=False):
    model = NeuralNet(device)
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


def compute_nus(v_mat, phi_cross_mat, betas, user_id):
    # computes Eq (14)
    # v_mat b X M X K
    # phi_cross_mat K X K
    # betas b X M X K

    k = user_id
    b, M, K = betas.shape

    temp = betas[:, :, k].view(b, M, 1)
    nu_mat = phi_cross_mat[:, k] * (torch.sqrt(v_mat) * (temp / betas))  # b X M X K

    return nu_mat

def individual_utility_computation(betas, mus, N, geta_d, T_p, T_c, v_mat, phi_cross_mat, target_user):
    # Eq (16) and (17)
    # v_mat b X M X K
    # phi_cross_mat K X K
    # betas b X M X K
    # mus b X M X K

    (b, M, K) = betas.shape
    k = target_user

    nu_mat_k = compute_nus(v_mat, phi_cross_mat, betas, k)  # Eq (14) b X M X K:  [nu]_{ik} for all i, k given
    temp_sum = 0
    for i in range(K):
        temp_sum += (1 / N) * (mus[:, :, i].view(b, 1, M) @ (betas[:, :, k:(k + 1)] * mus[:, :, i:(i + 1)])) + (
                nu_mat_k[:, :, i].view(b, 1, M) @ mus[:, :, i:(i + 1)]) ** 2

    temp_sum *= geta_d  # b X 1 X 1

    num = geta_d * (nu_mat_k[:, :, k].view(b, 1, M) @ mus[:, :, k:(k + 1)]) ** 2
    gamma = num / (temp_sum - num + (1 / N ** 2))
    SE = (1 - T_p / T_c) * torch.log(1 + gamma)  # b X 1 X 1
    SE = torch.squeeze(SE)
    return nu_mat_k, SE

def compute_smooth_min(SE_vec):
    # SE_vec is of dim either K 1 or b X K
    tau = 3
    SE_smooth_min = -(1 / tau) * torch.log((torch.exp(-tau * SE_vec)).mean(dim=-1))  # scalar or b X 1
    return SE_smooth_min

def utility_computation(betas, mus, N, geta_d, T_p, T_c, phi_cross_mat, v_mat, device):
    K = betas.shape[2]
    SE = torch.zeros((K,), device=device, requires_grad=False, dtype=torch.float32)
    for k in range(K):
        _, SE[k] = individual_utility_computation(betas, mus, N, geta_d, T_p, T_c, v_mat, phi_cross_mat, k)  # Eq (16)

    return [compute_smooth_min(SE), SE]