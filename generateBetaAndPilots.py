# -*- coding: utf-8 -*-
import torch
import os


def getUserConfig(areaWidth, areaHeight, numberOfUsers, device, forInsights=False, Tp=None):
    areaDims = torch.tensor(
        [areaWidth, areaHeight],
        device=device,
        requires_grad=False,
        dtype=torch.float32
    )

    torch.seed()

    assert not forInsights or Tp is not None, "Tp must not be None when forInsights is True"
    if not forInsights or Tp <= numberOfUsers:
        randVec = torch.rand(
                            (2, numberOfUsers),
                            device=device,
                            requires_grad=False,
                            dtype=torch.float32
                        ) - 0.5
        userConfig = torch.einsum('d,dm->md ', areaDims, randVec).to(device)
        return userConfig

    userList = []

    # 1. First Tp users — random as before
    randVec = torch.rand(
                            (2, Tp),
                            device=device,
                            requires_grad=False,
                            dtype=torch.float32
                        ) - 0.5
    baseUsers = torch.einsum('d,dm->md', areaDims, randVec)
    userList.append(baseUsers)

    # 2. Precompute epsilon in physical units
    epsilon = 0.05 * torch.mean(areaDims).item()

    # 3. Remaining users: alternating pattern
    remaining = numberOfUsers - Tp
    for i in range(remaining):
        if i % 2 == 0:
            # Even: nearby user around user i (from baseUsers)
            center = baseUsers[i]  # shape: (2,)

            # Sample uniformly from a disk of radius epsilon
            r = torch.sqrt(torch.rand(1, device=device)) * epsilon
            theta = 2 * torch.pi * torch.rand(1, device=device)
            offset = torch.cat((r * torch.cos(theta), r * torch.sin(theta)))
            newUser = (center + offset).unsqueeze(0)  # shape: (1, 2)
        else:
            # Odd: random user in full area
            randVec = torch.rand((2,), device=device) - 0.5
            newUser = (areaDims * randVec).unsqueeze(0)  # shape: (1, 2)

        userList.append(newUser)

    # 4. Concatenate all user positions
    userConfig = torch.cat(userList, dim=0).to(device)  # shape: (numberOfUsers, 2)

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

def dataGen(simulationParameters, systemParameters, sampleId, validationData=False,
            forInsights=False):
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
    if simulationParameters.minNumberOfUsersFlag:
        numberOfUsers = systemParameters.minNumberOfUsers

    Tp = systemParameters.Tp
    userConfig = getUserConfig(areaWidth, areaHeight, numberOfUsers, device,
                               forInsights=forInsights, Tp=Tp)  # get user positions
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
        pilotSequence = torch.randint(0, Tp, (numberOfUsers,))
    else:
        uniquePilotAllocation = torch.randperm(Tp)
        if forInsights:
            # Repeat the uniquePilotAllocation pattern deterministically
            repeatCount = (numberOfUsers + Tp - 1) // Tp  # ceil division
            fullPilotSequence = uniquePilotAllocation.repeat(repeatCount)[:numberOfUsers]
            pilotSequence = fullPilotSequence
        else:
            additionalNumOfUsers = numberOfUsers - Tp
            if additionalNumOfUsers <= 0:
                pilotSequence = uniquePilotAllocation[:numberOfUsers]
            else:
                reUsedPilotAllocation = torch.randint(0, Tp, (additionalNumOfUsers,))
                pilotSequence = torch.cat((uniquePilotAllocation, reUsedPilotAllocation), dim=0)

    m = {'betas': betas.to('cpu'), 'pilotSequence': pilotSequence}
    torch.save(m, os.path.join(filePath, f'betasSample{sampleId}.pt'))
    