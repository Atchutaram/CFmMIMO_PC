import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import math
import torch.nn.functional as F
from math import sqrt


class RootDataset(Dataset):
    def __init__(self, dataPath, phiOrth, numSamples, maxNumberOfUsers, PAD_CONST):
        self.path = dataPath
        _, _, files = next(os.walk(self.path))
        self.numSamples = min(len(list(filter(lambda k: 'betas' in k, files))), numSamples)
        self.phiOrth = phiOrth
        self.maxNumberOfUsers = maxNumberOfUsers
        self.PAD_CONST = PAD_CONST
        
    def __getitem__(self, index):
        betaFileName = f'betasSample{index}.pt'
        betaFilePath = os.path.join(self.path, betaFileName)
        m = torch.load(betaFilePath)
        betaOriginal = m['betas'].to(dtype=torch.float32)
        pilotSequence = m['pilotSequence'].to(dtype=torch.int32)

        phi = torch.index_select(self.phiOrth, 0, pilotSequence)
        phiCrossMat = torch.abs(phi.conj() @ phi.T)
        
        actualNumberOfUsers = phiCrossMat.size(-1)
        padUsers = self.maxNumberOfUsers - actualNumberOfUsers
        phiCrossMatPadded = F.pad(phiCrossMat, (0, padUsers, 0, padUsers), 'constant', 0)
        betaOriginalPadded = F.pad(betaOriginal, (0, padUsers, 0, 0), 'constant', self.PAD_CONST)
        
        betaTorch = torch.log(betaOriginalPadded)
        betaTorch = betaTorch.to(dtype=torch.float32)
        betaTorch = betaTorch.reshape(betaOriginalPadded.shape)

        return phiCrossMatPadded, betaTorch, betaOriginalPadded, actualNumberOfUsers

    def __len__(self):
        return self.numSamples


class CommonParameters:
    numSamples = 1
    batchSize = 1024
    numEpochs = 4*2
    M2Multiplier = 100

    learningRate = 1 / sqrt(5*M2Multiplier)
    
    # Params related to varying step size
    VARYING_STEP_SIZE = True
    trainingDataPath = ''

    @classmethod
    def preInt(cls, simulationParameters, systemParameters):
        cls.M = systemParameters.numberOfAccessPoints
        cls.K = systemParameters.maxNumberOfUsers
        
        cls.numSamples = simulationParameters.numberOfSamples
        cls.trainingDataPath = simulationParameters.dataFolder
        cls.validationDataPath = simulationParameters.validationDataFolder
        cls.scenario = simulationParameters.scenario
        
        warmupSteps = 4000
        scaleFactor = 1
        exp1 = -0.5
        exp2 = -1.5
        
        if (simulationParameters.operationMode == 2) or (cls.scenario > 3):
            cls.batchSize = 1  # for either large-scale systems or for testing mode
        
        
        if (simulationParameters.scenario == 0):
            cls.learningRate = 1 / sqrt(5*8)
        elif (simulationParameters.scenario == 1):
            pass
        elif (simulationParameters.scenario == 2):
            pass
        elif (simulationParameters.scenario == 3):
            pass
        else:
            raise('Invalid Scenario Configuration')

        cls.lambdaLr = lambda step: scaleFactor * min(
                                                            (step + 1) ** (exp1),
                                                            (step + 1) * warmupSteps ** (exp2)
                                                    )



class RootNet(pl.LightningModule):
    def __init__(self, systemParameters, grads):
        super(RootNet, self).__init__()
        self.save_hyperparameters()

        torch.seed()
        self.automatic_optimization = False
        
        self.InpDataset = RootDataset
        self.N = systemParameters.numberOfAntennas
        self.N_invRoot = 1/math.sqrt(self.N)
        self.systemParameters = systemParameters
        self.grads = grads
        self.relu = nn.ReLU()
        self.name = None
        self.maxNumberOfUsers = self.systemParameters.maxNumberOfUsers
        self.PAD_CONST = 6e-13
        
    
    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        phiCrossMat, betaTorch, betaOriginal, _ = batch

        opt.zero_grad()
        mus = self([betaTorch, phiCrossMat])

        with torch.no_grad():
            [mus_grads, utility] = self.grads(
                                                    betaOriginal,
                                                    mus,
                                                    self.device,
                                                    self.systemParameters,
                                                    phiCrossMat
                                            )
        
        self.manual_backward(mus, None, gradient=mus_grads)
        opt.step()
        if self.VARYING_STEP_SIZE:
            sch = self.lr_schedulers()
            sch.step()
        
        with torch.no_grad():
            loss = -utility.mean() # loss is negative of the utility
        tensorboardLogs = {'trainLoss': loss}
        self.log('trainLoss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss, 'log': tensorboardLogs}
    

    def validation_step(self, batch, batch_idx):
        phiCrossMat, betaTorch, betaOriginal, _ = batch

        mus = self([betaTorch, phiCrossMat])

        [_, utility] = self.grads(
                                        betaOriginal,
                                        mus,
                                        self.device,
                                        self.systemParameters,
                                        phiCrossMat
                                )
        loss = -utility.mean()
        self.log('valLoss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return {"valLoss": loss}

    def backward(self, loss, *args, **kwargs):
        mus=loss
        mus_grads= kwargs['gradient']
        B = mus_grads.shape[0]
        mus.backward((1/B)*mus_grads)

    def configure_optimizers(self):
        if self.VARYING_STEP_SIZE:
            optimizer = torch.optim.Adam(
                                            self.parameters(),
                                            lr=self.learningRate,
                                            betas=(0.9, 0.98),
                                            eps=1e-9
                                        )
            return [optimizer], [LambdaLR(optimizer, lr_lambda=self.lambdaLr)]
        else:
            optimizer = torch.optim.Adam(
                                            self.parameters(),
                                            lr=self.learningRate
                                        )
            return optimizer
    
    def train_dataloader(self):
        trainDataset = self.InpDataset(
                                            dataPath=self.dataPath,
                                            phiOrth=self.systemParameters.phiOrth,
                                            numSamples=self.numSamples,
                                            maxNumberOfUsers = self.maxNumberOfUsers,
                                            PAD_CONST = self.PAD_CONST,
                                    )
        trainLoader = DataLoader(dataset=trainDataset, batch_size=self.batchSize, shuffle=True)
        return trainLoader

    def val_dataloader(self):
        valDataset = self.InpDataset(
                                        dataPath=self.valDataPath,
                                        phiOrth=self.systemParameters.phiOrth,
                                        numSamples=self.numSamples,
                                        maxNumberOfUsers = self.maxNumberOfUsers,
                                        PAD_CONST = self.PAD_CONST,
                                    )
        valLoader = DataLoader(dataset=valDataset, batch_size=self.batchSize, shuffle=False)
        return valLoader
