import torch

from .utils import load_the_latest_model_and_params_if_exists
from .setup import CommonParameters
from channel_env.gradient_handler import grad_f
from power_control.utils import utility_computation


def project_to_s(y, N, device):
    # Eq (29)
    y_plus = y * (y > 0)
    y_norm = torch.norm(y_plus, dim=0)
    const = 1 / torch.sqrt(torch.tensor(N, requires_grad=False, device=device, dtype=torch.float32))
    y_max = (y_norm > const) * y_norm + (y_norm < const) * const
    mus = const * y_plus / y_max
    return mus

def test(test_sample, model_folder, device):
    import pickle
    model = load_the_latest_model_and_params_if_exists(model_folder, is_testing=True)
    sc = pickle.load(open(CommonParameters.sc_path, 'rb'))

    with torch.no_grad():
        test_sample = torch.log(test_sample)

        t_shape = test_sample.shape
        test_sample = test_sample.view((-1, CommonParameters.input_size)).to(device='cpu')
        beta_torch = sc.transform(test_sample)[0]
        beta_torch = torch.tensor(beta_torch, device=device, requires_grad=False, dtype=torch.float32).view(t_shape)
        model = model.to(device=device)
        mus_predicted = model(beta_torch)
        return mus_predicted

def epa(v_mat, device):
    v_mat = torch.squeeze(v_mat, dim=0)
    etaa = 1 / v_mat.sum(dim=1)
    etaa_outer = torch.outer(etaa, torch.ones((v_mat.shape[1],), device=device, requires_grad=False, dtype=torch.float32))
    mus = torch.sqrt(etaa_outer * v_mat)
    mus = torch.unsqueeze(mus, dim=0)
    return mus

def algo_one(betas, N, geta_d, T_p, T_c, phi_cross_mat, v_mat, device):
    Lf = 1 / 20
    mus_vec_old = torch.rand(betas.shape, device=device)  # random initialization
    t_old = torch.tensor(1, requires_grad=False, device=device, dtype=torch.float32)
    t_new = t_old
    alpha = 1 / (4 * Lf)
    mus_vec_new = mus_vec_old
    z = mus_vec_new

    for n in range(50):
        # print(n, u_new)
        y = mus_vec_new + t_old / t_new * (z - mus_vec_new) + (t_old - 1) / t_new * (mus_vec_new - mus_vec_old)
        y_grad = grad_f(betas, y, N, geta_d, T_p, T_c, phi_cross_mat, v_mat, device)[0]
        z = project_to_s(y + alpha * y_grad, N, device)

        grad_mus_new = grad_f(betas, mus_vec_new, N, geta_d, T_p, T_c, phi_cross_mat, v_mat, device)[0]
        v = project_to_s(mus_vec_new + alpha * grad_mus_new, N, device)

        mus_vec_old = mus_vec_new
        u_z = utility_computation(betas, z, N, geta_d, T_p, T_c, phi_cross_mat, v_mat, device)[0]
        u_v = utility_computation(betas, v, N, geta_d, T_p, T_c, phi_cross_mat, v_mat, device)[0]
        mus_vec_new = z if u_z > u_v else v

        t_old = t_new
        t_new = 0.5 * (torch.sqrt(4 * t_new ** 2 + 1) + 1)
    mus = mus_vec_new

    return mus

def get_power_control_coefficients(betas, N, geta_d, T_p, T_c, phi_cross_mat, v_mat, device):

    mus_epa = epa(v_mat, device)
    mus_one = algo_one(betas, N, geta_d, T_p, T_c, phi_cross_mat, v_mat, device)
    mus_CNN = test(betas, device)
    
    return mus_epa, mus_one, mus_CNN