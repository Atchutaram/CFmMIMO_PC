import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import math
import torch.nn.functional as F

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
    numEpochs = 4*4
    numEpochs = 1

    learningRate = 1e-4
    
    # Params related to varying step size
    VARYING_STEP_SIZE = True
    gamma = 0.75  # LR decay multiplication factor
    stepSize = 1 # for varying lr
    trainingDataPath = ''

    @classmethod
    def preInt(cls, simulationParameters, systemParameters):
        cls.M = systemParameters.numberOfAccessPoints
        cls.K = systemParameters.maxNumberOfUsers
        
        cls.numSamples = simulationParameters.numberOfSamples
        cls.trainingDataPath = simulationParameters.dataFolder
        cls.validationDataPath = simulationParameters.validationDataFolder
        cls.scenario = simulationParameters.scenario
        cls.dropout = 0
        
        if (simulationParameters.operationMode == 2) or (cls.scenario > 2):
            cls.batchSize = 1  # for either large-scale systems or for testing mode
        


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
        self.PAD_CONST = 6e-10
        
    
    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        phiCrossMat, betaTorch, betaOriginal, actualNumberOfUsers = batch

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
        
        with torch.no_grad():
            loss = -utility.mean() # loss is negative of the utility
        tensorboardLogs = {'trainLoss': loss}
        self.log('trainLoss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss, 'log': tensorboardLogs}
    

    def validation_step(self, batch, batch_idx):
        phiCrossMat, betaTorch, betaOriginal, actualNumberOfUsers = batch

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

    def on_train_epoch_end(self, *args, **kwargs):
        # We are using this method to manipulate the learning rate according to our needs
        skipLength = 1
        LL=7
        UL=10
        if self.VARYING_STEP_SIZE:
            sch = self.lr_schedulers()
            if LL<=self.current_epoch<=UL and (self.current_epoch % skipLength==0):
                sch.step()
            # print('\nEpoch end', sch.get_last_lr())
        else:
            # print('\nEpoch end', self.learningRate)
            pass

        

    def backward(self, loss, *args, **kwargs):
        mus=loss
        mus_grads= kwargs['gradient']
        B = mus_grads.shape[0]
        mus.backward((1/B)*mus_grads)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learningRate) 
        if self.VARYING_STEP_SIZE:
            return [optimizer], [StepLR(optimizer, step_size=self.stepSize, gamma=self.gamma)]
        else:
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
