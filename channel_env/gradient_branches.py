import os
import sys
import torch
from torch._C import device
from torch.serialization import save

from power_control.utils import individual_utility_computation



def grad_f_k(betas, mus, N, geta_d, k_input, i_input, nu_mat):
    # Eq (32)
    # betas B X M X K
    # mus B X M X K
    # nu_mat B X M X K
    [B, M, K] = betas.shape

    # temp1 and temp2 are correct, no need to change.
    temp1 = 2 * geta_d * nu_mat[:, :, i_input] * torch.squeeze(
        nu_mat[:, :, i_input].view(B, 1, M) @ mus[:, :, i_input:(i_input + 1)], 2)  # B X M
    temp2 = 2 * (geta_d / N) * betas[:, :, k_input] * mus[:, :, i_input]  # B X M

    if k_input == i_input:
        b_dash = temp1  # B X M
        c_dash = temp2  # B X M
    else:
        b_dash = 0 * nu_mat[:, :, i_input]  # B X M
        c_dash = temp1 + temp2  # B X M

    b = geta_d * (
        torch.squeeze(nu_mat[:, :, k_input].view(B, 1, M) @ mus[:, :, k_input:(k_input + 1)], 2)) ** 2  # B X 1
    b_plus_c = 1 / (N ** 2)
    for kk in range(K):
        b_plus_c += geta_d * (nu_mat[:, :, kk].view(B, 1, M) @ mus[:, :, kk:(kk + 1)]) ** 2 + (geta_d / N) * (
                mus[:, :, kk].view(B, 1, M) @ (
                betas[:, :, k_input:(k_input + 1)] * mus[:, :, kk:(kk + 1)]))  # B X 1 X 1

    b_plus_c = torch.squeeze(b_plus_c, 2)  # B X 1
    SE_dash = (b_dash + c_dash) / b_plus_c - c_dash / (b_plus_c - b)  # B X M

    return SE_dash

def branch_fn(betas, y, N, geta_d, T_p, T_c, v_mat, phi_cross_mat, k, tau, device):
    
    (nu_mat, SE) = individual_utility_computation(betas, y, N, geta_d, T_p, T_c, v_mat, phi_cross_mat, k)  # Eq (16)
    b, M, K = betas.shape
    num = torch.zeros((b, M, K), device=device, requires_grad=False, dtype=torch.float32)
    
    for i in range(K):
        SE_grad = grad_f_k(betas, y, N, geta_d, k, i, nu_mat)  # graf of SE_k w.r.t mu_i (eq (42) got this wrong)
        num += SE_grad * torch.exp(-tau * torch.unsqueeze(SE, dim=1))  # b X M
    
    return (num, SE)

def load_params(branch_id, device):
    pass

def save_outputs(num, SE):
    pass

if __name__ == '__main__':
    argv = sys.argv[1:]
    if not argv:
        print('Something went wrong!')
        sys.exit()

    inp_path = os.path.join('//data.triton.aalto.fi', 'work', 'kochark1', 'CFmMIMO_PC_LS', 'exhange_logs', 'grad_inps')
    out_path = os.path.join('//data.triton.aalto.fi', 'work', 'kochark1', 'CFmMIMO_PC_LS', 'exhange_logs', 'grads')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    branch_id = argv[0]

    betas, y, N, geta_d, T_p, T_c, v_mat, phi_cross_mat, k, tau = load_params(branch_id, device)
    num, SE = branch_fn(betas, y, N, geta_d, T_p, T_c, v_mat, phi_cross_mat, k, tau, device)
    save_outputs(num, SE)