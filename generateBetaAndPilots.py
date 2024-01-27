# -*- coding: utf-8 -*-
import torch
import os


def getUserConfig(areaWidth, areaHeight, numberOfUsers, device):
    areaDims = torch.tensor(
            [areaWidth, areaHeight],
            device=device,
            requires_grad=False,
            dtype=torch.float32
        )
    torch.seed()
    randVec = torch.rand(
                            (2, numberOfUsers),
                            device=device,
                            requires_grad=False,
                            dtype=torch.float32
                        ) - 0.5
    userConfig = torch.einsum('d,dm->md ', areaDims, randVec).to(device)
    return userConfig

def get_dMat(userConfig, apMinusRef):
    d2 = userConfig.view(-1, 1, 1, 2) - apMinusRef
    dMat, _ = torch.min((torch.sqrt(torch.einsum('kmtc->mkt', d2**2))), dim=-1)
    return dMat

def pathLossModel(L, d0, d1, log_d0, log_d1, dMat):
    log_dMat = torch.log10(dMat)
    PL0 = (-L - 15 * log_d1 - 20 * log_d0) * (dMat <= d0)
    PL1 = (-L - 15 * log_d1 - 20 * log_dMat) * (d0 < dMat) * (dMat < d1)
    PL2 = (-L - 35 * log_dMat) * (dMat >= d1)
    PL = PL0 + PL1 + PL2
    return PL

def getLSFs(L, d0, d1, log_d0, log_d1, sigma_sh, dMat, device):
    torch.seed()
    ZTemp = torch.normal(
                            mean=0,
                            std=sigma_sh,
                            size=dMat.shape,
                            device=device,
                            requires_grad=False,
                            dtype=torch.float32
                        )
    PL = pathLossModel(L, d0, d1, log_d0, log_d1, dMat) + ZTemp
    betas = 10 ** (PL / 10)
    return betas

def dataGen(simulationParameters, systemParameters, sampleId, validationData=False):
    if validationData:
        filePath = simulationParameters.validationDataFolder
    else:
        filePath = simulationParameters.dataFolder
    device = simulationParameters.device
    areaWidth = systemParameters.areaWidth
    areaHeight = systemParameters.areaHeight
    if not simulationParameters.varyingNumberOfUsersFlag:
        numberOfUsers = systemParameters.maxNumberOfUsers
    else:
        numberOfUsers = torch.randint(
                                            systemParameters.minNumberOfUsers,
                                            systemParameters.maxNumberOfUsers + 1,
                                            (1, )
                                    ).item()

    userConfig = getUserConfig(areaWidth, areaHeight, numberOfUsers, device)  # get user positions
    # distance mat for each pair of AP and user
    dMat = get_dMat(userConfig, systemParameters.apMinusRef)
    
    L = systemParameters.param_L
    d0 = systemParameters.d0
    d1 = systemParameters.d1
    log_d0 = systemParameters.log_d0
    log_d1 = systemParameters.log_d1
    sigma_sh = systemParameters.sigma_sh
    betas = getLSFs(L, d0, d1, log_d0, log_d1, sigma_sh, dMat, device)
    
    torch.seed()
    if simulationParameters.randomPilotsFlag:
        pilotSequence = torch.randint(0, systemParameters.Tp, (numberOfUsers,))
    else:
        additionalNumOfUsers = numberOfUsers - systemParameters.Tp
        uniquePilotAllocation = torch.randperm(systemParameters.Tp)
        if additionalNumOfUsers<=0:
            pilotSequence = uniquePilotAllocation[:numberOfUsers]
        else:
            reUsedPilotAllocation = torch.randint(0, systemParameters.Tp, (additionalNumOfUsers,))
            pilotSequence = torch.cat((uniquePilotAllocation, reUsedPilotAllocation), dim=0)
    

    # Save the RX data and original channel matrix.
    m = {'betas': betas.to('cpu'), 'pilotSequence': pilotSequence}
    torch.save(m, os.path.join(filePath, f'betasSample{sampleId}.pt'))
    