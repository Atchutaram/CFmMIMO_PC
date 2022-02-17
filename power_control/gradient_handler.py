import torch
import os

from .utils import compute_vmat, compute_smooth_min


def compute_num_k(betas, mus, N, zeta_d, T_p, T_c, v_mat, phi_cross_mat, k, tau):
    nu_mat_k = torch.einsum('k, bmk, bm -> bmk', phi_cross_mat[:, k], (torch.sqrt(v_mat) / betas), betas[:, :, k])

    nu_dot_mu = torch.einsum('bmk,bmk->bk', nu_mat_k, mus)  # B X K
    beta_k_dot_mus = torch.einsum('bm,bmk->bmk', betas[:, :, k], mus)
    b_vec = zeta_d * (nu_dot_mu)**2
    term3 = (zeta_d / N) * (torch.einsum('bmk,bmk->bk', mus, beta_k_dot_mus)) + b_vec
    b_plus_c = 1 / (N ** 2) + term3.sum(1)
    
    b = b_vec[:, k] # B
    gamma = b / (b_plus_c - b)
    SE = (1 - T_p / T_c) * torch.log(1 + gamma)  # b X 1 X 1
    
    temp1_batch = 2 * zeta_d * nu_mat_k * torch.unsqueeze(nu_dot_mu, 1) # B X M X K
    temp2_batch = 2 * (zeta_d / N) * beta_k_dot_mus # B X M X K

    b_dash_batch = torch.zeros(nu_mat_k.shape, device=nu_mat_k.device, requires_grad=False, dtype=torch.float32) # B X M X K
    c_dash_batch = temp1_batch + temp2_batch # B X M X K
    
    b_dash_batch[:,:, k] = temp1_batch[:,:,k]
    c_dash_batch[:,:, k] = temp2_batch[:,:,k]
    
    SE_grad = torch.einsum('bmk,b->bmk', (b_dash_batch+c_dash_batch), 1/b_plus_c) - torch.einsum('bmk,b->bmk', c_dash_batch, 1/(b_plus_c - b))
    num = torch.einsum('bmk, b -> bmk', SE_grad, torch.exp(-tau * SE))
    return num, SE


def grad_f(betas, mus, N, zeta_d, T_p, T_c, phi_cross_mat, v_mat, tau, device):
    # Eq (42)
    # v_mat b X M X K
    # phi_cross_mat K X K
    # betas b X M X K
    # y b X M X K
    [B, M, K] = betas.shape

    SE = torch.zeros((B, K), device=device, requires_grad=False, dtype=torch.float32)
    num = torch.zeros((B, M, K), device=device, requires_grad=False, dtype=torch.float32)

    for k in range(K):
        num_k, SE[:,k] = compute_num_k(betas, mus, N, zeta_d, T_p, T_c, v_mat, phi_cross_mat, k, tau)
        num += num_k

    den = (torch.exp(-tau * SE)).sum(dim=1)  # b X 1
    grad = num / den.view(-1, 1, 1)  # Eq (42) b X M X K
    return [grad, SE]

def grads(betas_in, mus_in, eta, slack_variable, device, system_parameters):
    with torch.no_grad():
        tau = system_parameters.tau
        v_mat = compute_vmat(betas_in, system_parameters.zeta_p, system_parameters.T_p, system_parameters.phi_cross_mat, device)  # Eq (5) b X M X K
        [mus_out, SE] = grad_f(betas_in, mus_in, system_parameters.number_of_antennas, system_parameters.zeta_d, system_parameters.T_p, system_parameters.T_c, system_parameters.phi_cross_mat, v_mat, tau, device)  # [b X M X K, b X K]
        
        temp = torch.unsqueeze(1 / (1 / system_parameters.number_of_antennas - (torch.norm(mus_in, dim=2)) ** 2 - slack_variable ** 2), -1)  # b X M X 1
        mus_out -= eta * mus_in * temp  # b X M X K
        grad_wrt_slack = - eta * slack_variable * temp.sum(dim=1)  # b X 1

        mus_out, grad_wrt_slack = [-1 * mus_out, -1 * grad_wrt_slack]  # do not merge with earlier equation; keep it different for readability.
        utility = compute_smooth_min(SE, tau)  # b X 1
        return [mus_out, grad_wrt_slack, utility]