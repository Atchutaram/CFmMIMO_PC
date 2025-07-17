import os
import torch
import sys

from utils.utils import handleDeletionAndCreation
from parameters.modes import OperatingModes




class SimulationParameters:
    # This class Maintains all the simulation parameter settings
    
    def __init__(self, args, insightsFolder=None):
        (
            root,
            simulationId,
            numberOfSamples,
            operatingMode,
            scenario,
            retain,
            resultsBase,
            randomPilotsFlag,
            varyingNumberOfUsersFlag,
            minNumberOfUsersFlag,
        ) = (
                args.root,
                args.simulationId,
                args.numberOfSamples,
                args.operatingMode,
                args.scenario,
                args.retain,
                args.resultsBase,
                args.randomPilotsFlag,
                args.varyingNumberOfUsersFlag,
                args.minNumberOfUsersFlag,
            )
        
        self.numberOfSamples = numberOfSamples
        self.validationNumberOfData = int(numberOfSamples * 0.25)
        self.operationMode = operatingMode
        self.modelFolder = f'modelsSc{scenario}'
        self.scenario = scenario
        self.randomPilotsFlag = randomPilotsFlag
        self.varyingNumberOfUsersFlag = varyingNumberOfUsersFlag
        self.minNumberOfUsersFlag = minNumberOfUsersFlag
        self.simulationId = simulationId
        
        if (torch.cuda.is_available() and (not (self.operationMode==OperatingModes.TESTING))):
            deviceTxt = "cuda"
        else:
            deviceTxt = "cpu"
        self.device = torch.device(deviceTxt)
        
        self.rootPath = root
        if not os.path.exists(self.rootPath):
            print(self.rootPath)
            print('rootPath failure!')
            sys.exit()

        if self.operationMode == OperatingModes.TRAINING:
            self.baseFolder = 'dataLogsTraining'
        else:
            self.baseFolder = 'dataLogsTesting'

        simIdName = f'simId{simulationId}'
        
        self.baseFolderPath = os.path.join(self.rootPath, simIdName, self.baseFolder)
        if resultsBase is None:
            self.resultsBase = os.path.join(self.rootPath, simIdName)
        else:
            self.resultsBase = os.path.join(resultsBase, simIdName)
        
        
        if self.operationMode == OperatingModes.INSIGHTS:
            if insightsFolder is None:
                print('dataFolder is expected as input in Operating mode 7')
                exit()
            self.baseFolderPath = os.path.join(insightsFolder, 'dataLogsInsights')
            self.resultsFolder = os.path.join(insightsFolder, 'results')
            self.plotFolder = os.path.join(insightsFolder, 'plots')

        if self.operationMode == OperatingModes.TESTING:
            handleDeletionAndCreation(self.resultsBase, forceRetain=True)
            
        self.modelFolderPath = self.resultsBase
            
        self.dataFolder = os.path.join(self.baseFolderPath, "betas")
        self.validationDataFolder = os.path.join(self.baseFolderPath, "betasVal")
        if operatingMode not in [OperatingModes.TRAINING, OperatingModes.INSIGHTS]:
            resTail = str(int(self.varyingNumberOfUsersFlag)) + str(int(self.randomPilotsFlag))
            if self.minNumberOfUsersFlag:
                resTail = 'minK'
            self.resultsFolder = os.path.join(self.resultsBase, "results_"+ resTail)
            self.plotFolder = os.path.join(self.resultsBase, "plots_" + resTail)
        

        if not self.operationMode == OperatingModes.PLOTTING_ONLY:
            if not os.path.exists(self.baseFolderPath):
                os.makedirs(self.baseFolderPath)

            handleDeletionAndCreation(self.dataFolder, self.numberOfSamples, retain)
            # The above function deletes and re-creates the folder only if retain=False.
            # If we request different number of data samples than existing, then retain fails.
            
            if not operatingMode==OperatingModes.TRAINING:
                handleDeletionAndCreation(self.resultsFolder)
                handleDeletionAndCreation(self.plotFolder, forceRetain= True)
            else:
                handleDeletionAndCreation(
                                            self.validationDataFolder,
                                            self.validationNumberOfData,
                                            retain
                                        )

        elif self.operationMode == OperatingModes.PLOTTING_ONLY:
            if not os.path.exists(self.resultsFolder) or len(os.listdir(self.resultsFolder)) == 0:
                print(f'Either {self.resultsFolder} folder is missing or empty')
                print('Run TESTING mode before running PLOTTING_ONLY mode!')
                sys.exit()
            
            handleDeletionAndCreation(self.plotFolder, forceRetain= True)

        
        if not os.path.exists(self.modelFolderPath):
            if self.operationMode == OperatingModes.TESTING:
                print(self.modelFolderPath)
                print('Train the neural network before testing!')
                sys.exit()
            os.makedirs(self.modelFolderPath)
        
    
    def handleModelSubFolders(self, modelsList):
        self.modelSubfolderPathDict = {}
        self.interimSubfolderPathDict = {}
        for modelName in modelsList:

            subfolderPath = os.path.join(self.modelFolderPath, modelName)
            self.modelSubfolderPathDict[modelName] = subfolderPath
            if not os.path.exists(subfolderPath):
                if self.operationMode == OperatingModes.TESTING:
                    print(subfolderPath)
                    print('Train the neural network before testing!')
                    sys.exit()