# -*- coding: utf-8 -*-
# import numpy as np
import torch
import os
import sys



def get_user_config(area_width, area_height, number_of_users, device):
    user_config = torch.empty([number_of_users, 2], device=device, dtype=torch.float32)
    for user in range(number_of_users):
        user_config[user] = torch.tensor([area_width, area_height], device=device, requires_grad=False, dtype=torch.float32) * (torch.rand((2,), device=device, requires_grad=False, dtype=torch.float32) - 0.5)
    return user_config

def get_d_mat(user_config, ap_config, area_width, area_height, device):
    d_mat = torch.zeros([len(ap_config), len(user_config)], device=device, requires_grad=False, dtype=torch.float32)
    D1 = area_width
    D2 = area_height
    ref = torch.tensor([[0, 0], [-D1, 0], [0, -D2], [D1, 0], [0, D2], [-D1, D2], [D1, -D2], [-D1, -D2], [D1, D2]], device=device, requires_grad=False, dtype=torch.float32)

    for ap_id, access_point in enumerate(ap_config):
        for user_id, user in enumerate(user_config):
            d_mat[ap_id, user_id] = torch.norm(user - (access_point - ref), dim=1).min()
    return d_mat

def path_loss_model(L, d_0, d_1, d_mat, device):
    L = torch.tensor(L, device=device, requires_grad=False, dtype=torch.float32)
    d_0 = torch.tensor(d_0, device=device, requires_grad=False, dtype=torch.float32)
    d_1 = torch.tensor(d_1, device=device, requires_grad=False, dtype=torch.float32)
    icassp_mode = True
    if icassp_mode:
        PL_0 = (-L - 15 * torch.log10(d_1) - 20 * torch.log10(d_0)) * (d_mat <= d_0)
        PL_1 = (-L - 15 * torch.log10(d_1) - 20 * torch.log10(d_mat)) * (d_0 < d_mat) * (d_mat < d_1)
        PL_2 = (-L - 35 * torch.log10(d_mat)) * (d_mat >= d_1)
        PL = PL_0 + PL_1 + PL_2
    else:
        # 3GPP model for a journal
        h_BS = 25
        h_UT = 1.5
        h_diff = h_BS - h_UT
        d_3D = torch.sqrt(d_mat ** 2 + h_diff ** 2)
        f = 3.4

        PL = 32.4 + 20 * torch.log10(f) + 30 * torch.log10(d_3D)

    return PL

def large_scale_fading_computing(L, d_0, d_1, sigma_sh, d_mat, device):
    Z_temp = torch.normal(mean=0, std=sigma_sh, size=d_mat.shape, device=device, requires_grad=False, dtype=torch.float32)
    PL = path_loss_model(L, d_0, d_1, d_mat, device=device) + Z_temp
    betas = 10 ** (PL / 10)
    return betas

def carefully_save_file(m, file):
    import time
    
    while True:
        try:
            torch.save(m, file)
        except:
            pass
        
        # time.sleep(2)
        if os.path.exists(file):
            break

def data_gen(simulation_parameters, sample_id):
    from .params import SystemParameters
    
    scenario = simulation_parameters.scenario
    file_path = simulation_parameters.data_folder
    device = simulation_parameters.device

    if scenario==1:
        inp_param_D = 1
        inp_number_of_users = 20
        inp_access_point_density = 100
        system_parameters = SystemParameters(simulation_parameters, inp_param_D, inp_number_of_users, inp_access_point_density)
    elif scenario==2:
        inp_param_D = 1
        inp_number_of_users = 500
        inp_access_point_density = 2000
        system_parameters = SystemParameters(simulation_parameters, inp_param_D, inp_number_of_users, inp_access_point_density)
    else:
        system_parameters = SystemParameters(simulation_parameters)


    if os.path.exists(os.path.join(file_path, f'betas_sample{sample_id}.pt')):
        print('Delete data folder folder')
        sys.exit()

    
    area_width = system_parameters.area_width
    area_height = system_parameters.area_height
    number_of_users = system_parameters.number_of_users

    user_config = get_user_config(area_width, area_height, number_of_users, device)  # get user positions
    ap_config = system_parameters.AP_configs # AP positions

    d_mat = get_d_mat(user_config, ap_config, area_width, area_height, device)  # distance matrix for each pair of AP and user
    L = system_parameters.param_L
    d_0 = system_parameters.d_0
    d_1 = system_parameters.d_1
    sigma_sh = system_parameters.sigma_sh
    betas = large_scale_fading_computing(L, d_0, d_1, sigma_sh, d_mat, device)
    

    # Save the RX data and original channel matrix.
    m = {'betas': betas.to('cpu'), }
    carefully_save_file(m, os.path.join(file_path, f'betas_sample{sample_id}.pt'))
    