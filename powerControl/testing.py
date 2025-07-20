import os
import torch
import time
from tqdm import tqdm

from .utils import compute_v_mat, utilityComputation
from .gradientHandler import grads, grad_f
from .models.utils import (loadTheLatestModelAndParamsIfExists,
                           deploy,
                           initializeHyperParams,
                           dumpAttention)
from utils.visualization import (performancePlotter, consolidatedPlotter, localPlotEditor,
                                 visualizeAttentions)


def project2s(y, const):
    # Eq (29)
    epsilon = 1e-8
    yPlus = y * (y > 0)
    yNorm = torch.unsqueeze(torch.sqrt(torch.einsum('bmk, bmk -> bm', yPlus, yPlus)), -1)
    yMax = torch.clamp(yNorm, min = const)
    mus = const * yPlus / (yMax + epsilon)
    return mus


def epa(vMat, device):
    vMat = torch.squeeze(vMat, dim=0)
    etaa = 1 / vMat.sum(dim=1)
    etaaOuter = torch.outer(
                                etaa,
                                torch.ones(
                                                (vMat.shape[1],),
                                                device=device,
                                                requires_grad=False,
                                                dtype=torch.float32
                                        )
                            )
    mus = torch.sqrt(etaaOuter * vMat)
    mus = torch.unsqueeze(mus, dim=0)
    return mus


def apgAlgo(betas, N, zeta_d, Tp, Tc, phiCrossMat, vMat, tau, device):
    import math
    torch.seed()
    
    # random initialization
    musVecOld = torch.rand(betas.shape, requires_grad=False, device=device, dtype=torch.float32)

    tOld = 0
    tNew = 1
    musVecNew = musVecOld
    z = musVecNew
    torch.seed()
    y = musVecNew\
            + 0.01*torch.rand(betas.shape, requires_grad=False, device=device, dtype=torch.float32)
    v = y * 0
    rho = 0.8
    delta = 1e-5

    const =  1 / math.sqrt(N)
    epsilon = 1e-10

    for _ in range(30):

        s = z - y
        r = grad_f(betas, z, N, zeta_d, Tp, Tc, phiCrossMat, vMat, tau, device)[0]\
                - grad_f(betas, y, N, zeta_d, Tp, Tc, phiCrossMat, vMat, tau, device)[0]
        denTemp = torch.dot(s.flatten(), r.flatten()) + epsilon
        alpha_y = torch.dot(s.flatten(), s.flatten()) / denTemp

        s = v - musVecOld
        r = grad_f(betas, v, N, zeta_d, Tp, Tc, phiCrossMat, vMat, tau, device)[0]\
                - grad_f(betas, musVecOld, N, zeta_d, Tp, Tc, phiCrossMat, vMat, tau, device)[0]
        denTemp = torch.dot(s.flatten(), r.flatten()) + epsilon
        alpha_mu = torch.dot(s.flatten(), s.flatten()) / denTemp

        alpha_y = torch.abs(alpha_y)
        alpha_mu = torch.abs(alpha_mu)
        y = musVecNew\
                + (tOld / tNew) * (z - musVecNew)\
                + ((tOld - 1) / tNew) * (musVecNew - musVecOld)

        while 1:
            z = project2s(
                                y + alpha_y * grad_f(
                                                        betas,
                                                        y,
                                                        N,
                                                        zeta_d,
                                                        Tp,
                                                        Tc,
                                                        phiCrossMat,
                                                        vMat,
                                                        tau,
                                                        device
                                                    )[0],
                                const
                        )
            alpha_y = rho * alpha_y
            u_z, _ = utilityComputation(betas, z, N, zeta_d, Tp, Tc, phiCrossMat, vMat, tau, device)
            u_y, _ = utilityComputation(betas, y, N, zeta_d, Tp, Tc, phiCrossMat, vMat, tau, device)
            deltaDiff = delta * torch.dot((z - y).flatten(), (z - y).flatten())
            
            if alpha_y < epsilon:
                break

            if math.isnan(u_z + u_y + alpha_y) or math.isinf(u_z + u_y + alpha_y):
                import sys
                print('APG error in loop1')
                print(u_z, u_y, alpha_y)
                sys.exit()

            if u_z >= (u_y + deltaDiff):
                break

        while 1:
            v = project2s(
                                musVecNew + alpha_mu * grad_f(
                                                                    betas,
                                                                    musVecNew,
                                                                    N,
                                                                    zeta_d,
                                                                    Tp,
                                                                    Tc,
                                                                    phiCrossMat,
                                                                    vMat,
                                                                    tau,
                                                                    device
                                                                )[0],
                                const
                        )
            alpha_mu = rho * alpha_mu
            u_v, _ = utilityComputation(betas, v, N, zeta_d, Tp, Tc, phiCrossMat, vMat, tau, device)
            u_mu, _ = utilityComputation(
                                            betas,
                                            musVecNew,
                                            N,
                                            zeta_d,
                                            Tp,
                                            Tc,
                                            phiCrossMat,
                                            vMat,
                                            tau,
                                            device
                                        )
            deltaDiff = delta * torch.dot((v - musVecNew).flatten(), (v - musVecNew).flatten())
            
            if alpha_mu < epsilon:
                break
                
            if math.isnan(u_v + u_mu + alpha_mu) or math.isinf(u_v + u_mu + alpha_mu):
                import sys
                print('APG error in loop2')
                print(u_v, u_mu, alpha_mu)
                sys.exit()

            if u_v >= (u_mu + deltaDiff):
                break
        musVecOld = musVecNew
        musVecNew = z if u_z > u_v else v

        tOld = tNew
        tNew = 0.5 * (math.sqrt(4 * tNew ** 2 + 1) + 1)
    

    return musVecNew, max(u_z, u_v)


def runPowerControlAlgos(simulationParameters, systemParameters, algoList, models, sampleId):
    device = torch.device('cpu')

    filePathAndName = os.path.join(simulationParameters.dataFolder, f'betasSample{sampleId}.pt')
    m = torch.load(filePathAndName)

    betas = m['betas'].to(dtype=torch.float32, device=device)
    betas = torch.unsqueeze(betas, 0)

    pilotSequence = m['pilotSequence']

    N = systemParameters.numberOfAntennas
    zeta_d = systemParameters.zeta_d
    zeta_p = systemParameters.zeta_p
    Tp = systemParameters.Tp
    Tc = systemParameters.Tc

    phi = torch.index_select(systemParameters.phiOrth, 0, pilotSequence)
    phiCrossMat = torch.abs(phi.conj() @ phi.T).to(dtype=torch.float32, device=device)
    phiCrossMat = torch.unsqueeze(phiCrossMat**2, dim=0)
    tau = systemParameters.tau

    vMat = compute_v_mat(betas, zeta_p, Tp, phiCrossMat)

    latencyDict = {}
    seDict = {}
    for algoName in algoList:
        
        timeThen = time.perf_counter()
        if algoName == 'EPA':
            mus = epa(vMat, device)
        elif algoName == 'APG':
            mus, _ = apgAlgo(betas, N, zeta_d, Tp, Tc, phiCrossMat, vMat, tau, device)
        else:
            modelName = algoName  # this algo is deep learning algo
            mus = deploy(models[modelName], betas, phiCrossMat, modelName, device)
        
        timeNow = time.perf_counter()
        latencyDict[algoName] = round(timeNow - timeThen, 6)
        
        _, seDict[algoName] = utilityComputation(
                                                betas,
                                                mus,
                                                N,
                                                zeta_d,
                                                Tp,
                                                Tc,
                                                phiCrossMat,
                                                vMat,
                                                tau,
                                                device
                                            )
    return seDict, latencyDict


def saveLatency(resultPath, latency):
    m = {'latency': latency, }
    torch.save(m, os.path.join(resultPath, f'latency.pt'))


def loadLatency(resultPath):
    return torch.load(os.path.join(resultPath, f'latency.pt'))['latency']


def setupAndLoadDeepLearningModels(modelsToRun, simulationParameters, systemParameters):
    modelFolderDict = simulationParameters.modelSubfolderPathDict
    
    
    models = {}
    for modelName in modelsToRun:
        initializeHyperParams(modelName, simulationParameters, systemParameters)
        models[modelName] = loadTheLatestModelAndParamsIfExists(
                                                                    modelName,
                                                                    modelFolderDict[modelName],
                                                                    systemParameters,
                                                                    grads,
                                                                    isTesting=True
                                                                )
    
    return models


def testAndPlot(simulationParameters, systemParameters, plottingOnly):
    device = torch.device('cpu')  # Need to force this. We do not want to test in GPU.
    algoList = ['EPA', 'APG', ]
    modelsList = systemParameters.models  # deep learning models

    algoList += modelsList
    
    resultsPath = simulationParameters.resultsFolder
    if plottingOnly:
        avgLatency = loadLatency(resultsPath)
    else:
        LOG_CONVERSION_CONST = torch.log2(torch.exp(torch.scalar_tensor(1))).to(device=device)
        models = setupAndLoadDeepLearningModels(modelsList, simulationParameters, systemParameters)
        numberOfSamples = simulationParameters.numberOfSamples

        avgLatency = {}
        for algoName in algoList:
            avgLatency[algoName] = 0
        
        for sampleId in tqdm(range(numberOfSamples)):
            
            seDict, latencyDict = runPowerControlAlgos(
                                                            simulationParameters,
                                                            systemParameters,
                                                            algoList,
                                                            models,
                                                            sampleId
                                                    )
            for algoName in algoList:
                resultSample = seDict[algoName]*LOG_CONVERSION_CONST  # nat/sec/Hz to bits/sec/Hz
                avgLatency[algoName] += (1/numberOfSamples)*latencyDict[algoName]

                resultSampleDict = {'resultSample': resultSample, }
                filePath = os.path.join(resultsPath, f'{algoName}ResultsSample{sampleId}.pt')
                torch.save(resultSampleDict, filePath)

        saveLatency(resultsPath, avgLatency)

    performancePlotter(
                            resultsPath,
                            algoList,
                            simulationParameters.plotFolder,
                            simulationParameters.scenario
                        )
    print(avgLatency)
    return avgLatency

def consolidatePlot(figIdx, resultsFolders, algoLists, tags, tagsForNonML, plotFolder):
    consolidatedPlotter(figIdx, resultsFolders, algoLists, tags, tagsForNonML, plotFolder)

def localPlotEditing(figIdx, plotFolder, outputFolder):
    localPlotEditor(figIdx, plotFolder, outputFolder)

def visualizeInsights(simulationParameters, systemParameters):
    modelsList = ['PAPC']

    algoList = modelsList
    
    models = setupAndLoadDeepLearningModels(modelsList, simulationParameters, systemParameters)
    numberOfSamples = simulationParameters.numberOfSamples
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for sampleId in range(numberOfSamples):
        filePathAndName = os.path.join(simulationParameters.dataFolder, f'betasSample{sampleId}.pt')
        dumpPath = os.path.join(simulationParameters.resultsFolder, f'attnMat{sampleId}.pt')
        m = torch.load(filePathAndName)

        betas = m['betas'].to(dtype=torch.float32, device=device)
        betas = torch.unsqueeze(betas, 0)

        pilotSequence = m['pilotSequence']

        phi = torch.index_select(systemParameters.phiOrth, 0, pilotSequence)
        phiCrossMat = torch.abs(phi.conj() @ phi.T).to(dtype=torch.float32, device=device)
        phiCrossMat = torch.unsqueeze(phiCrossMat**2, dim=0)
        
        for algoName in algoList:
            
            modelName = algoName  # this algo is deep learning algo
            dumpAttention(models[modelName], betas, phiCrossMat, modelName, device, dumpPath)
    
    avg = 0
    for sampleId in range(numberOfSamples):
        x = torch.load(os.path.join(simulationParameters.resultsFolder, f'attnMat{sampleId}.pt'))
        avg += x/numberOfSamples
    
    avg = avg.squeeze(0)
    visualizeAttentions(simulationParameters.plotFolder, avg)