import torch

from .utils import compute_v_mat, computeSmoothMin


def compute_num_k(betas, mus, N, zeta_d, Tp, Tc, vMat, phiCrossMat, k, tau):
    nuMat_k = torch.einsum(
                                'bk, bmk, bm -> bmk',
                                phiCrossMat[:, :, k],
                                (torch.sqrt(vMat) / betas),
                                betas[:, :, k]
                            )
    
    nuDotMu = torch.einsum('bmk,bmk->bk', nuMat_k, mus)  # B X K
    beta_k_dot_mus = torch.einsum('bm,bmk->bmk', betas[:, :, k], mus)
    bVec = zeta_d * nuDotMu**2
    term3 = (zeta_d / N) * (torch.einsum('bmk,bmk->bk', mus, beta_k_dot_mus)) + bVec
    b_plus_c = 1 / (N ** 2) + term3.sum(1)
    
    b = bVec[:, k] # B
    gamma = b / (b_plus_c - b)
    se = (1 - Tp / Tc) * torch.log(1 + gamma)  # b X 1 X 1
    
    temp1Batch = 2 * zeta_d * nuMat_k * torch.unsqueeze(nuDotMu, 1) # B X M X K
    temp2Batch = 2 * (zeta_d / N) * beta_k_dot_mus # B X M X K

    bDashBatch = torch.zeros(
                                    nuMat_k.shape,
                                    device=nuMat_k.device,
                                    requires_grad=False,
                                    dtype=torch.float32
                                ) # B X M X K
    cDashBatch = temp1Batch + temp2Batch # B X M X K
    
    bDashBatch[:,:, k] = temp1Batch[:,:,k]
    cDashBatch[:,:, k] = temp2Batch[:,:,k]
    
    seGrad = torch.einsum('bmk,b->bmk', (bDashBatch+cDashBatch), 1/b_plus_c)\
              - torch.einsum('bmk,b->bmk', cDashBatch, 1/(b_plus_c - b))
    num = torch.einsum('bmk, b -> bmk', seGrad, torch.exp(-tau * se))
    return num, se


def grad_f(betas, mus, N, zeta_d, Tp, Tc, phiCrossMat, vMat, tau, device):
    # Eq (42)
    # v_mat b X M X K
    # phi_cross_mat b x K X K
    # betas b X M X K
    # y b X M X K
    [B, M, K] = betas.shape

    SE = torch.zeros((B, K), device=device, requires_grad=False, dtype=torch.float32)
    num = torch.zeros((B, M, K), device=device, requires_grad=False, dtype=torch.float32)

    for k in range(K):
        num_k, SE[:,k] = compute_num_k(
                                            betas,
                                            mus,
                                            N,
                                            zeta_d,
                                            Tp,
                                            Tc,
                                            vMat,
                                            phiCrossMat,
                                            k,
                                            tau
                                        )
        num += num_k

    den = (torch.exp(-tau * SE)).sum(dim=1)  # b X 1
    grad = num / den.view(-1, 1, 1)  # Eq (42) b X M X K
    return [grad, SE]

def grads(betasIn, musIn, device, systemParameters, phiCrossMat):
    with torch.no_grad():
        tau = systemParameters.tau
        
        # Eq (5) b X M X K
        vMat = compute_v_mat(betasIn, systemParameters.zeta_p, systemParameters.Tp, phiCrossMat)
        [musOut, se] = grad_f(
                                    betasIn,
                                    musIn,
                                    systemParameters.numberOfAntennas,
                                    systemParameters.zeta_d,
                                    systemParameters.Tp,
                                    systemParameters.Tc,
                                    phiCrossMat,
                                    vMat,
                                    tau,
                                    device
                                )  # [b X M X K, b X K]
        
        musOut= -musOut  # Reason: gradient of loss = - gradient of utility
        utility = computeSmoothMin(se, tau)  # b X 1
        return [musOut, utility]