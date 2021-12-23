import torch
from .params import SystemParameters



def save_inputs(beta_original, mus):
    pass

def delete_inputs():
    pass

def compute_num():
    # load the output values of arryjob and compute num
    pass

def wait_and_check():
    # Wait till the arrays jobs that compute the gradient is completed.
    pass

def grads_num(beta_original, mus):
    save_inputs(beta_original, mus)
    wait_and_check()
    num = compute_num()
    delete_inputs()
    return num

def compute_vmat(betas, geta_p, T_p, phi_cross_mat, device):
    # computes Eq (5)
    # phi_cross_mat K X K
    # betas b X M X K

    den = torch.ones(betas.shape, device=device, requires_grad=False, dtype=torch.float32) + geta_p * T_p * (
            betas @ (phi_cross_mat ** 2))
    v_mat = (geta_p * T_p * (betas ** 2)) / den

    return v_mat

def compute_smooth_min(SE_vec):
    # SE_vec is of dim either K 1 or b X K
    tau = 3
    SE_smooth_min = -(1 / tau) * torch.log((torch.exp(-tau * SE_vec)).mean(dim=-1))  # scalar or b X 1
    return SE_smooth_min

def grad_f(betas, y, N, geta_d, T_p, T_c, phi_cross_mat, v_mat, device):
    # Eq (42)
    # v_mat b X M X K
    # phi_cross_mat K X K
    # betas b X M X K
    # y b X M X K
    tau = 3

    [B, M, K] = betas.shape

    SE = torch.zeros((B, K), device=device, requires_grad=False, dtype=torch.float32)
    # num = torch.zeros((B, M, K), device=device, requires_grad=False, dtype=torch.float32)

    num = grads_num(betas, y)

    den = (torch.exp(-tau * SE)).sum(dim=1)  # b X 1
    grad = num / den.view(-1, 1, 1)  # Eq (42) b X M X K
    return [grad, SE]

def grads(betas_in, mus_in, eta, slack_variable, device):
    with torch.no_grad():
        v_mat = compute_vmat(betas_in, SystemParameters.zeta_p, SystemParameters.T_p, SystemParameters.phi_cross_mat, device)  # Eq (5) b X M X K
        temp = torch.unsqueeze(1 / (1 / SystemParameters.number_of_antennas - (torch.norm(mus_in, dim=2)) ** 2 - slack_variable ** 2), 2)  # b X M X 1
        [mus_out, SE] = grad_f(betas_in, mus_in, SystemParameters.number_of_antennas, SystemParameters.zeta_d, SystemParameters.T_p, SystemParameters.T_c, SystemParameters.phi_cross_mat, v_mat, device)  # [b X M X K, b X K]
        mus_out -= eta * mus_in * temp  # b X M X K
        grad_wrt_slack = - eta * slack_variable * temp.sum(dim=1)  # b X 1

        mus_out, grad_wrt_slack = [-1 * mus_out, -1 * grad_wrt_slack]  # do not merge with earlier equation; keep it different for readability.
        utility = compute_smooth_min(SE)  # b X 1
        return [mus_out, grad_wrt_slack, utility]