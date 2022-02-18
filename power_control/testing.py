import os
import torch
import time

from .utils import compute_vmat, utility_computation
from .gradient_handler import grads, grad_f
from .models.utils import load_the_latest_model_and_params_if_exists, deploy, model_test_setup
from utils.visualization import performance_plotter



def project_to_s(y, const):
    # Eq (29)
    y_plus = y * (y > 0)
    y_norm = torch.unsqueeze(torch.sqrt(torch.einsum('bmk, bmk -> bm', y_plus, y_plus)), -1)
    y_max = torch.clamp(y_norm, min=const)
    mus = const * y_plus / y_max
    return mus

def epa(v_mat, device):
    v_mat = torch.squeeze(v_mat, dim=0)
    etaa = 1 / v_mat.sum(dim=1)
    etaa_outer = torch.outer(etaa, torch.ones((v_mat.shape[1],), device=device, requires_grad=False, dtype=torch.float32))
    mus = torch.sqrt(etaa_outer * v_mat)
    mus = torch.unsqueeze(mus, dim=0)
    return mus


def ref_algo_one(betas, N, zeta_d, T_p, T_c, phi_cross_mat, v_mat, tau, device):
    import math
    Lf = 1 / 20
    mus_vec_old = torch.rand(betas.shape, requires_grad=False, device=device, dtype=torch.float32)  # random initialization
    t_old = 1
    t_new = t_old
    alpha = 1 / (4 * Lf)
    mus_vec_new = mus_vec_old
    z = mus_vec_new

    const =  1 / math.sqrt(N)

    for _ in range(50):
        y = mus_vec_new + t_old / t_new * (z - mus_vec_new) + (t_old - 1) / t_new * (mus_vec_new - mus_vec_old)
        y_grad, _ = grad_f(betas, y, N, zeta_d, T_p, T_c, phi_cross_mat, v_mat, tau, device)
        z = project_to_s(y + alpha * y_grad, const)

        grad_mus_new, _ = grad_f(betas, mus_vec_new, N, zeta_d, T_p, T_c, phi_cross_mat, v_mat, tau, device)
        v = project_to_s(mus_vec_new + alpha * grad_mus_new, const)

        mus_vec_old = mus_vec_new
        u_z, _ = utility_computation(betas, z, N, zeta_d, T_p, T_c, phi_cross_mat, v_mat, tau, device)
        u_v, _ = utility_computation(betas, v, N, zeta_d, T_p, T_c, phi_cross_mat, v_mat, tau, device)
        mus_vec_new = z if u_z > u_v else v

        t_old = t_new
        t_new = 0.5 * (math.sqrt(4 * t_new ** 2 + 1) + 1)
    return mus_vec_new, max(u_z, u_v).item()


def ref_algo_two(betas, N, zeta_d, T_p, T_c, phi_cross_mat, v_mat, tau, device):
    import math
    mus_vec_old = torch.rand(betas.shape, requires_grad=False, device=device, dtype=torch.float32)  # random initialization

    t_old = 0
    t_new = 1
    mus_vec_new = mus_vec_old
    z = mus_vec_new
    y = mus_vec_new + 0.01*torch.rand(betas.shape, requires_grad=False, device=device, dtype=torch.float32)
    v = y * 0
    rho = 0.8
    delta = 1e-5

    const =  1 / math.sqrt(N)

    for _ in range(30):

        s = z - y
        r = grad_f(betas, z, N, zeta_d, T_p, T_c, phi_cross_mat, v_mat, tau, device)[0] - grad_f(betas, y, N, zeta_d, T_p, T_c, phi_cross_mat, v_mat, tau, device)[0]
        alpha_y = torch.dot(s.flatten(), s.flatten()) / torch.dot(s.flatten(), r.flatten())

        s = v - mus_vec_old
        r = grad_f(betas, v, N, zeta_d, T_p, T_c, phi_cross_mat, v_mat, tau, device)[0] - grad_f(betas, mus_vec_old, N, zeta_d, T_p, T_c, phi_cross_mat, v_mat, tau, device)[0]
        alpha_mu = torch.dot(s.flatten(), s.flatten()) / torch.dot(s.flatten(), r.flatten())

        alpha_y = torch.abs(alpha_y)
        alpha_mu = torch.abs(alpha_mu)
        y = mus_vec_new + (t_old / t_new) * (z - mus_vec_new) + ((t_old - 1) / t_new) * (mus_vec_new - mus_vec_old)

        count = 200
        while 1:
            z = project_to_s(y + alpha_y * grad_f(betas, y, N, zeta_d, T_p, T_c, phi_cross_mat, v_mat, tau, device)[0], const)
            alpha_y = rho * alpha_y
            u_z, _ = utility_computation(betas, z, N, zeta_d, T_p, T_c, phi_cross_mat, v_mat, tau, device)
            u_y, _ = utility_computation(betas, y, N, zeta_d, T_p, T_c, phi_cross_mat, v_mat, tau, device)
            delta_diff = delta * torch.dot((z - y).flatten(), (z - y).flatten())
            
            count += -1
            if count<0:
                import sys
                print('algo_two took too long')
                sys.exit()

            if u_z >= (u_y + delta_diff):
                break

        count = 200
        while 1:
            v = project_to_s(mus_vec_new + alpha_mu * grad_f(betas, mus_vec_new, N, zeta_d, T_p, T_c, phi_cross_mat, v_mat, tau, device)[0], const)
            alpha_mu = rho * alpha_mu
            u_v, _ = utility_computation(betas, v, N, zeta_d, T_p, T_c, phi_cross_mat, v_mat, tau, device)
            u_mu, _ = utility_computation(betas, mus_vec_new, N, zeta_d, T_p, T_c, phi_cross_mat, v_mat, tau, device)
            delta_diff = delta * torch.dot((v - mus_vec_new).flatten(), (v - mus_vec_new).flatten())
            
            count += -1
            if count<0:
                import sys
                print('algo_two took too long')
                sys.exit()

            if u_v >= (u_mu + delta_diff):
                break
        mus_vec_old = mus_vec_new
        mus_vec_new = z if u_z > u_v else v

        t_old = t_new
        t_new = 0.5 * (math.sqrt(4 * t_new ** 2 + 1) + 1)
    

    return mus_vec_new, max(u_z, u_v)

def run_power_control_algos(simulation_parameters, system_parameters, algo_list, models, sample_id):
    device = simulation_parameters.device

    file_path_and_name = os.path.join(simulation_parameters.data_folder, f'betas_sample{sample_id}.pt')
    betas = torch.load(file_path_and_name)['betas'].to(dtype=torch.float32, device=device)
    betas = torch.unsqueeze(betas, 0)

    N = system_parameters.number_of_antennas
    zeta_d = system_parameters.zeta_d
    zeta_p = system_parameters.zeta_p
    T_p = system_parameters.T_p
    T_c = system_parameters.T_c
    phi_cross_mat = system_parameters.phi_cross_mat
    tau = system_parameters.tau

    v_mat = compute_vmat(betas, zeta_p, T_p, phi_cross_mat, device)

    latency = {}
    SE = {}
    if 'epa' in algo_list:
        algo_name = 'epa'

        time_then = time.perf_counter()
        mus = epa(v_mat, device)
        time_now = time.perf_counter()
        latency[algo_name] = round(time_now - time_then, 6)
        
        _, SE[algo_name] = utility_computation(betas, mus, N, zeta_d, T_p, T_c, phi_cross_mat, v_mat, tau, device)

    if 'ref_algo_one' in algo_list:
        algo_name = 'ref_algo_one'

        time_then = time.perf_counter()
        mus, _ = ref_algo_one(betas, N, zeta_d, T_p, T_c, phi_cross_mat, v_mat, tau, device)
        time_now = time.perf_counter()
        latency[algo_name] = round(time_now - time_then, 6)

        _, SE[algo_name] = utility_computation(betas, mus, N, zeta_d, T_p, T_c, phi_cross_mat, v_mat, tau, device)
        
        

    if 'ref_algo_two' in algo_list:
        algo_name = 'ref_algo_two'

        time_then = time.perf_counter()
        mus, _ = ref_algo_two(betas, N, zeta_d, T_p, T_c, phi_cross_mat, v_mat, tau, device)
        time_now = time.perf_counter()
        latency[algo_name] = round(time_now - time_then, 6)

        _, SE[algo_name] = utility_computation(betas, mus, N, zeta_d, T_p, T_c, phi_cross_mat, v_mat, tau, device)


    if 'CNN' in algo_list:
        algo_name = 'CNN'
        model_name = 'CNN'

        time_then = time.perf_counter()
        mus = deploy(models[model_name], betas, model_name, device)
        time_now = time.perf_counter()
        latency[algo_name] = round(time_now - time_then, 6)

        _, SE[algo_name] = utility_computation(betas, mus, N, zeta_d, T_p, T_c, phi_cross_mat, v_mat, tau, device)


    if 'FCN' in algo_list:
        algo_name = 'FCN'
        model_name = 'FCN'

        time_then = time.perf_counter()
        mus = deploy(models[model_name], betas, model_name, device)
        time_now = time.perf_counter()
        latency[algo_name] = round(time_now - time_then, 6)

        _, SE[algo_name] = utility_computation(betas, mus, N, zeta_d, T_p, T_c, phi_cross_mat, v_mat, tau, device)
    
    return SE, latency

def save_latency(result_path, latency):
    m = {'latency': latency, }
    torch.save(m, os.path.join(result_path, f'latency.pt'))

def load_latency(result_path):
    return torch.load(os.path.join(result_path, f'latency.pt'))['latency']

def setup_and_load_deep_learning_models(models_to_run, simulation_parameters, system_parameters):
    model_folder_dict = simulation_parameters.model_subfolder_path_dict
    interm_folder_dict = simulation_parameters.interm_subfolder_path_dict
    device = simulation_parameters.device
    
    number_of_access_points = system_parameters.number_of_access_points
    number_of_users = system_parameters.number_of_users
    scenario = simulation_parameters.scenario
    
    models = {}
    for model_name in models_to_run:
        model_test_setup(number_of_access_points, number_of_users, scenario, model_name)
        models[model_name] = load_the_latest_model_and_params_if_exists(model_name, model_folder_dict[model_name], interm_folder_dict[model_name], system_parameters, grads, device, is_testing=True)
    
    return models

def test_and_plot(simulation_parameters, system_parameters, plotting_only):
    algo_list = ['epa', 'ref_algo_one', 'ref_algo_two', ]
    models_list = system_parameters.models_list  # deep learning models

    algo_list += models_list
    
    algos_to_skip = ['ref_algo_one', ]  # this is to facilitate preventing unwanted list of testing algorithms from running
    for algo_name in algos_to_skip:
        if algo_name in algo_list:
            algo_list.remove(algo_name)
    
    models_to_run = [value for value in algo_list if value in models_list]  # find deep learning models that are not skipped
    
    results_path = simulation_parameters.results_folder
    if plotting_only:
        avg_latency = load_latency(results_path)
    else:
        CONST = torch.log2(torch.exp(torch.scalar_tensor(1))).to(device=simulation_parameters.device)
        models = setup_and_load_deep_learning_models(models_to_run, simulation_parameters, system_parameters)
        number_of_samples = simulation_parameters.number_of_samples

        avg_latency = {}
        for algo_name in algo_list:
            avg_latency[algo_name] = 0
        
        for sample_id in range(number_of_samples):
            
            SE, latency = run_power_control_algos(simulation_parameters, system_parameters, algo_list, models, sample_id)
        
            for algo_name in algo_list:
                result_sample = SE[algo_name]*CONST  # nat/sec/Hz to bits/sec/Hz
                avg_latency[algo_name] += (1/number_of_samples)*latency[algo_name]
                
                m = {'result_sample': result_sample, }
                torch.save(m, os.path.join(results_path, f'{algo_name}_results_sample{sample_id}.pt'))

        save_latency(results_path, avg_latency)

    performance_plotter(results_path, algo_list, simulation_parameters.plot_folder, simulation_parameters.scenario)
    print(avg_latency)