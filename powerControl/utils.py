import torch
import math


def compute_v_mat(betas, zeta_p, Tp, phiCrossMat):
    # computes Eq (5)
    # phiCrossMat b x K X K
    # betas b X M X K

    den = torch.ones(betas.shape, device=betas.device, requires_grad=False, dtype=torch.float32)\
            + zeta_p * Tp * (betas @ (phiCrossMat ** 2))
    vMat = (zeta_p * Tp * (betas ** 2)) / den

    return vMat


def individualUtilityComputation(betas, mus, N, zeta_d, Tp, Tc, vMat, phiCrossMat, targetUser):
    # Eq (16) and (17)
    # vMat b X M X K
    # phiCrossMat K X K
    # betas b X M X K
    # mus b X M X K

    k = targetUser
 
    nu_mat_k = torch.einsum(
                                'bk, bmk, bm -> bmk',
                                phiCrossMat[:, :, k],
                                (torch.sqrt(vMat) / betas),
                                betas[:, :, k]
                            )

    nu_dot_mu = torch.einsum('bmk,bmk->bk', nu_mat_k, mus)  # B X K
    beta_k_dot_mus = torch.einsum('bm,bmk->bmk', betas[:, :, k], mus)
    bVec = zeta_d * (nu_dot_mu)**2
    term3 = (zeta_d / N) * (torch.einsum('bmk,bmk->bk', mus, beta_k_dot_mus)) + bVec
    b_plus_c = 1 / (N ** 2) + term3.sum(1)
    
    b = bVec[:, k] # B
    gamma = b / (b_plus_c - b)
    SE = (1 - Tp / Tc) * torch.log(1 + gamma)  # b X 1 X 1
    return nu_mat_k, SE


def computeSmoothMin(seVec, tau):
    # seVec is of dim either K 1 or b X K
    seSmoothMin = -(1 / tau) * torch.log((torch.exp(-tau * seVec)).mean(dim=-1))
    return seSmoothMin

def utilityComputation(betas, mus, N, geta_d, Tp, Tc, phiCrossMat, vMat, tau, device):
    K = betas.shape[-1]
    se = torch.zeros((K,), device=device, requires_grad=False, dtype=torch.float32)
    for k in range(K):
        _, se[k] = individualUtilityComputation(
                                                    betas,
                                                    mus,
                                                    N,
                                                    geta_d,
                                                    Tp,
                                                    Tc,
                                                    vMat,
                                                    phiCrossMat,
                                                    k
                                                )  # Eq (16)
    
    if math.isnan(se.sum().item()):
        print('Something went wrong! The utility is nan')
        print(betas.sum().item(), mus.sum().item())

    return [computeSmoothMin(se, tau), se]