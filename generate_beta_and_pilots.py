# -*- coding: utf-8 -*-
import torch
import os


def get_user_config(area_width, area_height, number_of_users, device):
    area_dims = torch.tensor([area_width, area_height], device=device, requires_grad=False, dtype=torch.float32)
    torch.seed()
    rand_vec = (torch.rand((2,number_of_users), device=device, requires_grad=False, dtype=torch.float32) - 0.5)
    user_config = torch.einsum('d,dm->md ', area_dims, rand_vec).to(device)
    return user_config

def get_d_mat(user_config, ap_minus_ref):
    d2 = user_config.view(-1, 1, 1, 2) - ap_minus_ref
    d_mat, _ = torch.min((torch.sqrt(torch.einsum('kmtc->mkt', d2**2))), dim=-1)
    return d_mat

def path_loss_model(L, d_0, d_1, log_d_0, log_d_1, d_mat):
    log_d_mat = torch.log10(d_mat)
    PL_0 = (-L - 15 * log_d_1 - 20 * log_d_0) * (d_mat <= d_0)
    PL_1 = (-L - 15 * log_d_1 - 20 * log_d_mat) * (d_0 < d_mat) * (d_mat < d_1)
    PL_2 = (-L - 35 * log_d_mat) * (d_mat >= d_1)
    PL = PL_0 + PL_1 + PL_2
    return PL

def large_scale_fading_computing(L, d_0, d_1, log_d_0, log_d_1, sigma_sh, d_mat, device):
    torch.seed()
    Z_temp = torch.normal(mean=0, std=sigma_sh, size=d_mat.shape, device=device, requires_grad=False, dtype=torch.float32)
    PL = path_loss_model(L, d_0, d_1, log_d_0, log_d_1, d_mat) + Z_temp
    betas = 10 ** (PL / 10)
    return betas

def data_gen(simulation_parameters, system_parameters, sample_id, validation_data=False):
    if validation_data:
        file_path = simulation_parameters.validation_data_folder
    else:
        file_path = simulation_parameters.data_folder
    device = simulation_parameters.device
    area_width = system_parameters.area_width
    area_height = system_parameters.area_height
    number_of_users = system_parameters.number_of_users

    user_config = get_user_config(area_width, area_height, number_of_users, device)  # get user positions
    d_mat = get_d_mat(user_config, system_parameters.ap_minus_ref)  # distance matrix for each pair of AP and user
    
    L = system_parameters.param_L
    d_0 = system_parameters.d_0
    d_1 = system_parameters.d_1
    log_d_0 = system_parameters.log_d_0
    log_d_1 = system_parameters.log_d_1
    sigma_sh = system_parameters.sigma_sh
    betas = large_scale_fading_computing(L, d_0, d_1, log_d_0, log_d_1, sigma_sh, d_mat, device)

    if simulation_parameters.orthogonality_flag:
        pilot_sequence = torch.arange(0, number_of_users)
    else:
        torch.seed()
        pilot_sequence = torch.randint(0, system_parameters.T_p, [number_of_users])
    

    # Save the RX data and original channel matrix.
    m = {'betas': betas.to('cpu'), 'pilot_sequence': pilot_sequence}
    torch.save(m, os.path.join(file_path, f'betas_sample{sample_id}.pt'))
    