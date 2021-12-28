import torch
import os
import sys
import warnings

from simulation_parameters.sim_params import OperatingModes


def compute_laplace_mat(ap_positions_list, device):
    N = ap_positions_list.shape[0]
    adjacency_matrix = torch.zeros((N, N), requires_grad=False, dtype=torch.float32, device=device)
    for ap_id1, ap_ref in enumerate(ap_positions_list):
        for ap_id2, ap_target in enumerate(ap_positions_list):
            if ap_id2 <= ap_id1:
                continue
            adjacency_matrix[ap_id1, ap_id2] = torch.norm(ap_target - ap_ref)
    adjacency_matrix = adjacency_matrix + adjacency_matrix.T

    adjacency_matrix_final = adjacency_matrix * 0
    number_of_adjacent_branches = 4
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for row_id, row in enumerate(adjacency_matrix):
            second_lowest = torch.sort(row).values[number_of_adjacent_branches - 1]
            adjacency_matrix_final[row_id, :] = (1 / row) * (row <= second_lowest)
            adjacency_matrix_final[row_id, row_id] = 0
    degree_matrix = torch.diag(adjacency_matrix.sum(dim=0))
    laplace_matrix = degree_matrix - adjacency_matrix_final

    return laplace_matrix


def folder_string_generator(number_of_users, number_of_access_points):
    return f'{number_of_users:04d}_users_{number_of_access_points:04d}_APs'



class SystemParameters:
    def __init__(self, simulation_parameters, param_D = 1, number_of_users = 20, access_point_density = 100):
        # Fixed parameters
        self.param_L = 140.715087  # dB
        self.d_0 = 0.01  # in Km
        self.d_1 = 0.05  # in Km
        self.sigma_sh = 8  # in dB
        self.band_width = 20e6  # in Hz
        self.noise_figure = 9  # in dB
        self.zeta_d = 1  # in W
        self.zeta_p = 0.2  # in W

        self.No_Hz = -173.975
        self.total_noise_power = 10 ** ((self.No_Hz - 30) / 10) * self.band_width * 10 ** (self.noise_figure / 10)
        self.zeta_d /= self.total_noise_power
        self.zeta_p /= self.total_noise_power

        self.T_p = 20
        self.T_c = 200

        # Scenario based parameters (Following is a set of parameters for a default scenario). These can be over written in simulate_Communications module
        self.number_of_antennas = 1  # N
        self.param_D = torch.tensor(param_D, requires_grad=False, device=simulation_parameters.device, dtype=torch.float32)  # D
        self.number_of_users = number_of_users
        self.access_point_density = access_point_density

        # Initialize sub-folder list
        self.sub_folder_list = []
        self.sub_folder_AP_user_mapping = []

        # to set other derived parameters of the scenario (special or default)
        self.number_of_access_points = int(access_point_density * self.param_D.item())
        self.area_width = torch.sqrt(self.param_D)  # in Km
        self.area_height = self.area_width

        random_mat = torch.normal(0, 1, (self.T_p, self.T_p))
        u, _, _ = torch.linalg.svd(random_mat)
        phi_orth = u
        phi = torch.zeros([self.number_of_users, self.T_p], requires_grad=False, device=simulation_parameters.device, dtype=torch.float32) * (1 + 1j)
        for k in range(self.number_of_users):
            i = torch.randint(0, self.T_p, [1]).item()
            phi[k] = phi_orth[:][i]
        self.phi_cross_mat = torch.abs(phi.conj() @ phi.T)

        self.AP_configs = []
        self.laplace_matrices = []
        ap_positions_list = torch.zeros([self.number_of_access_points, 2], requires_grad=False, device=simulation_parameters.device, dtype=torch.float32)
        for access_point_id in range(self.number_of_access_points):
            torch.manual_seed(seed=0 * self.number_of_access_points + access_point_id)
            ap_positions_list[access_point_id, :] = torch.tensor([self.area_width, self.area_height], device='cpu', requires_grad=False, dtype=torch.float32) * (torch.rand((2,), device='cpu', requires_grad=False, dtype=torch.float32)-0.5)
        self.laplace_matrices.append(compute_laplace_mat(ap_positions_list, simulation_parameters.device))
        self.AP_configs.append(ap_positions_list)

        self.AP_configs = torch.cat(self.AP_configs).to(simulation_parameters.device)

        # Initialize sub-folder list
        self.sub_folder_list = []
        self.sub_folder_AP_user_mapping = []
        if simulation_parameters.operation_mode == OperatingModes.plotting_only:
            if not os.path.exists(simulation_parameters.results_folder):
                print('Not results. Switch to Testing mode and fixed = True!')
                sys.exit()
        