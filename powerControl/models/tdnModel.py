import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import pickle
from sklearn.preprocessing import StandardScaler

from .rootModel import Mode, RootDataset, CommonParameters, RootNet


MODEL_NAME = 'TDN'


# Hyper-parameters
class HyperParameters(CommonParameters):
    sc = StandardScaler()
    scPath = os.path.join(os.getcwd(), f'{MODEL_NAME}sc.pkl')
    trainingDataPath = ''

    @classmethod
    def intialize(cls, simulationParameters, systemParameters, isTestMode):
        
        cls.preInt(simulationParameters, systemParameters)
        cls.hid = 50*cls.K
        
        if isTestMode:
            cls.batchSize = 1
            return
        
        if cls.scenario == 1:
            cls.batchSize = 8 * 2
        else:
            cls.batchSize = 1
        
        trainDataset = cls.InpDataSet(
                                            dataPath=cls.trainingDataPath,
                                            normalizer=cls.sc,
                                            mode=Mode.preProcessing,
                                            numSamples=cls.numSamples,
                                            device=torch.device('cpu')
                                    )
        trainLoader = DataLoader(dataset=trainDataset, batchSize=1, shuffle=False)
        
        for beta in trainLoader:
            with torch.no_grad():
                cls.sc.partial_fit(beta)

        pickle.dump(cls.sc, open(cls.scPath, 'wb'))  # saving sc for loading later in testing phase
        print(f'{cls.scPath} dumped!')
    
    

class NeuralNet(RootNet):
    def __init__(self, device, systemParameters, grads):
        super(NeuralNet, self).__init__(device, systemParameters, grads)
        
        self.numSamples = HyperParameters.numSamples
        self.numEpochs = HyperParameters.numEpochs
        self.eta = HyperParameters.eta
        self.dataPath = HyperParameters.trainingDataPath
        self.normalizer = HyperParameters.sc
        self.batchSize = HyperParameters.batchSize
        self.learningRate = HyperParameters.learningRate
        self.VARYING_STEP_SIZE = HyperParameters.VARYING_STEP_SIZE
        self.gamma = HyperParameters.gamma
        self.stepSize = HyperParameters.stepSize
        K = HyperParameters.K
        M = HyperParameters.M
        hid = HyperParameters.hid
        
        self.fcns = nn.ModuleList()
        for _ in range(M):
            self.fcns.append(
                nn.Sequential(
                    nn.Linear(K, hid),
                    nn.ReLU(),
                    nn.Linear(hid, hid),
                    nn.ReLU(),
                    nn.Linear(hid, K),
                    )
            )

        self.name = MODEL_NAME
        self.to(self.device)


    def forward(self, x):
        x, _ = x
        x = torch.unsqueeze(x, 1)
        decoded = []
        for m, FCN in enumerate(self.fcns):
            decodedTemp = FCN(x[:, 0, m, :])
            
            # so max final output after torch.exp is always between 0 and 1.
            # This conditioning helps regularization.
            decodedTemp = -self.relu(decodedTemp)
            decodedTemp = (1/self.systemParameters.numberOfAntennas) * torch.exp(decodedTemp)
            decoded.append(torch.unsqueeze(decodedTemp, 0))
        
        decoded = torch.transpose(torch.cat(decoded), 0, 1).to(device=self.device)*1e-1
        return decoded