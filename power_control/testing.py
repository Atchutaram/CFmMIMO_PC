import os
import torch
import time

from .utils import compute_vmat, utility_computation, load_the_latest_model_and_params_if_exists
from .gradient_handler import grad_f
from .deploy import deploy
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
    return mus_vec_new


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

        while 1:
            z = project_to_s(y + alpha_y * grad_f(betas, y, N, zeta_d, T_p, T_c, phi_cross_mat, v_mat, tau, device)[0], const)
            alpha_y = rho * alpha_y
            u_z, _ = utility_computation(betas, z, N, zeta_d, T_p, T_c, phi_cross_mat, v_mat, tau, device)
            u_y, _ = utility_computation(betas, y, N, zeta_d, T_p, T_c, phi_cross_mat, v_mat, tau, device)
            delta_diff = delta * torch.dot((z - y).flatten(), (z - y).flatten())

            if u_z >= (u_y + delta_diff):
                break

        while 1:
            v = project_to_s(mus_vec_new + alpha_mu * grad_f(betas, mus_vec_new, N, zeta_d, T_p, T_c, phi_cross_mat, v_mat, tau, device)[0], const)
            alpha_mu = rho * alpha_mu
            u_v, _ = utility_computation(betas, v, N, zeta_d, T_p, T_c, phi_cross_mat, v_mat, tau, device)
            u_mu, _ = utility_computation(betas, mus_vec_new, N, zeta_d, T_p, T_c, phi_cross_mat, v_mat, tau, device)
            delta_diff = delta * torch.dot((v - mus_vec_new).flatten(), (v - mus_vec_new).flatten())
            if u_v >= (u_mu + delta_diff):
                break
        mus_vec_old = mus_vec_new
        mus_vec_new = z if u_z > u_v else v

        t_old = t_new
        t_new = 0.5 * (math.sqrt(4 * t_new ** 2 + 1) + 1)

    return mus_vec_new

def get_power_control_coefficients(model, betas, N, zeta_d, T_p, T_c, phi_cross_mat, v_mat, tau, algo_list, device):

    mus_list = []
    latency_list = []
    if 'epa' in algo_list:
        time_then = time.perf_counter()
        mus_epa = epa(v_mat, device)
        time_now = time.perf_counter()
        latency_epa = round(time_now - time_then, 6)
        mus_list.append(mus_epa)
        latency_list.append(latency_epa)

    if 'ref_algo_one' in algo_list:
        time_then = time.perf_counter()
        mus_one = ref_algo_one(betas, N, zeta_d, T_p, T_c, phi_cross_mat, v_mat, tau, device)
        time_now = time.perf_counter()
        latency_one = round(time_now - time_then, 6)
        mus_list.append(mus_one)
        latency_list.append(latency_one)

    if 'ref_algo_two' in algo_list:
        time_then = time.perf_counter()
        mus_two = ref_algo_two(betas, N, zeta_d, T_p, T_c, phi_cross_mat, v_mat, tau, device)
        time_now = time.perf_counter()
        latency_two = round(time_now - time_then, 6)
        mus_list.append(mus_two)
        latency_list.append(latency_two)

    if 'CNN' in algo_list:
        time_then = time.perf_counter()
        mus_CNN = deploy(model, betas, device)
        time_now = time.perf_counter()
        latency_CNN = round(time_now - time_then, 6)
        mus_list.append(mus_CNN)
        latency_list.append(latency_CNN)
    
    latency_tensor = torch.tensor(latency_list, requires_grad=False, device=device, dtype=torch.float32)
    return mus_list, latency_tensor

def save_latency(result_path, latency):
    m = {'latency': latency, }
    torch.save(m, os.path.join(result_path, f'latency.pt'))

def load_latency(result_path):
    return torch.load(os.path.join(result_path, f'latency.pt'))['latency']

def test_and_plot(simulation_parameters, system_parameters, plotting_only):
    algo_list = ['epa', 'ref_algo_one', 'ref_algo_two', 'CNN']
    skip_list = [False, True, False, False, ]  # skip ref_algo_one
    final_algo_list = []

    for index, skip_flag in enumerate(skip_list):
        if not skip_flag:
            final_algo_list.append(algo_list[index])

    if plotting_only:
        avg_latency = load_latency(simulation_parameters.results_folder)
    else:
        file_path = simulation_parameters.data_folder
        result_path = simulation_parameters.results_folder
        model_folder_path = simulation_parameters.model_folder_path
        avg_latency = 0
        device = simulation_parameters.device

        from .models.nn_setup import CommonParameters

        CommonParameters.test_setup(system_parameters.number_of_access_points, system_parameters.number_of_users, simulation_parameters.scenario)
        model = load_the_latest_model_and_params_if_exists(model_folder_path, device, system_parameters, simulation_parameters.interm_folder, is_testing=True)
        model = model.to(device=device)
        
        N = system_parameters.number_of_antennas
        zeta_d = system_parameters.zeta_d
        zeta_p = system_parameters.zeta_p
        T_p = system_parameters.T_p
        T_c = system_parameters.T_c
        phi_cross_mat = system_parameters.phi_cross_mat
        tau = system_parameters.tau


        for sample_id in range(simulation_parameters.number_of_samples):
            file_path_and_name = os.path.join(file_path, f'betas_sample{sample_id}.pt')
            betas = torch.load(file_path_and_name)['betas'].to(dtype=torch.float32, device=device)
            betas = torch.unsqueeze(betas, 0)

            v_mat = compute_vmat(betas, zeta_p, T_p, phi_cross_mat, device)  # Eq (5)
            mus_list, latency_tensor = get_power_control_coefficients(model, betas, N, zeta_d, T_p, T_c, phi_cross_mat, v_mat, tau, final_algo_list, device)

            
            for mus, algo_name in zip(mus_list, final_algo_list):
                _, SE_full = utility_computation(betas, mus, N, zeta_d, T_p, T_c, phi_cross_mat, v_mat, tau, device)
                result_sample = SE_full*torch.log2(torch.exp(torch.scalar_tensor(1)))  # nat/sec/Hz to bits/sec/Hz
                m = {'result_sample': result_sample, }
                torch.save(m, os.path.join(result_path, f'{algo_name}_results_sample{sample_id}.pt'))
            
            avg_latency += (1/simulation_parameters.number_of_samples)*latency_tensor
        
        save_latency(simulation_parameters.results_folder, avg_latency)

    performance_plotter(simulation_parameters.results_folder, final_algo_list, simulation_parameters.plot_folder, simulation_parameters.scenario)
    print(avg_latency.tolist())