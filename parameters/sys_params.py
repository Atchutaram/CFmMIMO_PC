import torch
import warnings


def compute_laplace_mat(ap_positions_list, device):
    M = ap_positions_list.shape[0]
    adjacency_matrix = torch.zeros((M, M), requires_grad=False, dtype=torch.float32, device=device)
    for ap_id1, ap_ref in enumerate(ap_positions_list):
        for ap_id2, ap_target in enumerate(ap_positions_list):
            if ap_id2 <= ap_id1:
                continue
            adjacency_matrix[ap_id1, ap_id2] = torch.linalg.norm(ap_target - ap_ref)
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


class SystemParameters:
    def __init__(self, simulation_parameters, param_D, number_of_users, access_point_density):
        self.param_L = torch.tensor(140.715087, device=simulation_parameters.device, requires_grad=False, dtype=torch.float32)
        self.d_0 = torch.tensor(0.01, device=simulation_parameters.device, requires_grad=False, dtype=torch.float32)
        self.d_1 = torch.tensor(0.05, device=simulation_parameters.device, requires_grad=False, dtype=torch.float32)
        self.sigma_sh = 8  # in dB
        self.band_width = 20e6  # in Hz
        self.noise_figure = 9  # in dB
        self.zeta_d = 1  # in W
        self.zeta_p = 0.2  # in W
        self.log_d_0 = torch.log10(self.d_0)
        self.log_d_1 = torch.log10(self.d_1)
        
        self.tau = 3
        

        self.No_Hz = -173.975
        self.total_noise_power = 10 ** ((self.No_Hz - 30) / 10) * self.band_width * 10 ** (self.noise_figure / 10)
        self.zeta_d /= self.total_noise_power
        self.zeta_p /= self.total_noise_power

        self.T_p = 20
        self.T_c = 200

        
        self.number_of_antennas = 1  # N
        self.param_D = torch.tensor(param_D, requires_grad=False, device=simulation_parameters.device, dtype=torch.float32)  # D
        self.number_of_users = number_of_users  # K
        self.access_point_density = access_point_density

        self.number_of_access_points = int(access_point_density * self.param_D.item())  # M
        self.area_width = torch.sqrt(self.param_D)  # in Km
        self.area_height = self.area_width

        random_mat = torch.normal(0, 1, (self.T_p, self.T_p))
        u, _, _ = torch.linalg.svd(random_mat)
        phi_orth = u
        
        column_indices = torch.randint(0, self.T_p, [self.number_of_users])
        phi = torch.index_select(phi_orth, 0, column_indices)
        self.phi_cross_mat = torch.abs(phi.conj() @ phi.T).to(simulation_parameters.device)

        area_dims = torch.tensor([self.area_width, self.area_height], device=simulation_parameters.device, requires_grad=False, dtype=torch.float32)
        
        torch.manual_seed(seed=0)
        rand_vec = torch.rand((2, self.number_of_access_points), device=simulation_parameters.device, requires_grad=False, dtype=torch.float32)-0.5  # 2 X M
        
        AP_positions = torch.einsum('d,dm->md ', area_dims, rand_vec).to(simulation_parameters.device)
        self.laplace_matrix = compute_laplace_mat(AP_positions, simulation_parameters.device) # currently unused but do not delete

        D1 = self.area_width
        D2 = self.area_height
        ref_list = [[0, 0], [-D1, 0], [0, -D2], [D1, 0], [0, D2], [-D1, D2], [D1, -D2], [-D1, -D2], [D1, D2]]
        ref = torch.tensor(ref_list, device=simulation_parameters.device, requires_grad=False, dtype=torch.float32)
        
        self.ap_minus_ref = AP_positions.view(-1, 1, 2)-ref
        