import time
import os

from parameters.modes import OperatingModes
from utils.handleInputArgs import Args
from utils.utils import cleanFolders, logSystemInfoAndLatency

defaultNumberOfSamples = 50
testingNumberOfSamples = 2000

# Handling command-line arguments
args = Args(defaultNumberOfSamples, )
if args.clean: cleanFolders()
args.preProcessArgs(testingNumberOfSamples, )


print("""\nWelcome to the Cell Free massive MIMO power control code.
Try 'python cellFreeMassMimoPowCtrl.py -h' to learn about passing optional command line arguments.
\n""")

from parameters.simParams import SimulationParameters
from parameters.sysParams import SystemParameters

from generateBetaAndPilots import dataGen

from powerControl.learning import train
from powerControl.testing import testAndPlot


def dataGenAndTest(args):
    
    simulationParameters = SimulationParameters(args)
    systemParameters = SystemParameters(simulationParameters)

    if not os.listdir(simulationParameters.dataFolder):
        for sampleId in range(args.numberOfSamples):
            dataGen(simulationParameters, systemParameters, sampleId)

    testAndPlot(simulationParameters, systemParameters, plottingOnly=False)


# Preparing the root directory for the logs.
args.setRootDir()

if __name__ == '__main__':
    start = time.perf_counter()

    if (not (args.operatingMode == OperatingModes.CONSOL) and
    not (args.operatingMode == OperatingModes.LOCAL)):
        simulationParameters = SimulationParameters(args)
        systemParameters = SystemParameters(simulationParameters)

        # Generating train & validation or test data.
        if not (simulationParameters.operationMode == OperatingModes.PLOTTING_ONLY):
            if not os.listdir(simulationParameters.dataFolder):
                timeThen = time.perf_counter()
                
                for sampleId in range(args.numberOfSamples):
                    # Generates Train/Test data
                    dataGen(simulationParameters, systemParameters, sampleId)
                    
                    if ((simulationParameters.operationMode == OperatingModes.TRAINING)
                        and (sampleId < simulationParameters.validationNumberOfData)):
                            # Generates Validation data
                            dataGen(
                                simulationParameters,
                                systemParameters,
                                sampleId,
                                validationData=True
                                )
                
                timeNow = time.perf_counter()
                print(f'Finished data generation in {round(timeNow - timeThen, 2)} second(s)')

        # Training and/or test the power control algorithms.
        if simulationParameters.operationMode==OperatingModes.TRAINING:
            
            timeThen = time.perf_counter()
            train(simulationParameters, systemParameters)
            timeNow = time.perf_counter()
            print(f'Finished training in {round(timeNow - timeThen, 2)} second(s)')
            
            if args.allModeFlag:
                args.setOperatingMode(OperatingModes.TESTING)
                args.setNumberOfSamples(testingNumberOfSamples)
                
                simulationParameters = SimulationParameters(args)
                systemParameters = SystemParameters(simulationParameters)

                if not os.listdir(simulationParameters.dataFolder):
                    for sampleId in range(args.numberOfSamples):
                        dataGen(simulationParameters, systemParameters, sampleId)

                testAndPlot(simulationParameters, systemParameters, plottingOnly=False)

        elif simulationParameters.operationMode==OperatingModes.TESTING:
            avgLatency = testAndPlot(simulationParameters, systemParameters, plottingOnly=False)
            logSystemInfoAndLatency(simulationParameters, avgLatency)

        elif simulationParameters.operationMode==OperatingModes.PLOTTING_ONLY:
            testAndPlot(simulationParameters, systemParameters, plottingOnly=True)
    elif args.operatingMode == OperatingModes.CONSOL:
        from utils.utils import handleDeletionAndCreation
        from powerControl.testing import consolidatePlot
        scenarioMapping = lambda simId: 0 if simId in range(0, 3) else (simId - 2)
        
        plotFolderBase = os.path.join(args.root, 'consolidatedResults')
        handleDeletionAndCreation(plotFolderBase, retain=False)
        args.operatingMode = OperatingModes.PLOTTING_ONLY
        
        figIdx = 1
        resultsFolders = []
        algoLists = []
        tags = [
                    'Sc. 0 - 100K Samples',
                    'Sc. 0 - 1M Samples',
                    'Sc. 0 - 12M Samples',
                ]
        tagsForNonML = [
                    'Sc. 0',
                ]
        args.varyingNumberOfUsersFlag = False
        args.randomPilotsFlag = False
        for loopId, simId in enumerate(range(3)):
            args.simulationId = simId
            args.scenario = scenarioMapping(simId)
            simulationParameters = SimulationParameters(args)
            systemParameters = SystemParameters(simulationParameters)
            resultsFolders.append(simulationParameters.resultsFolder)
            algoList = systemParameters.models
            if loopId==0:
                algoList.append('EPA')
                algoList.append('APG')
            algoLists.append(algoList)
        
        plotFolder = os.path.join(plotFolderBase, f'Fig{figIdx}')
        handleDeletionAndCreation(plotFolder, retain=False)
        consolidatePlot(figIdx, resultsFolders, algoLists, tags, tagsForNonML, plotFolder)
        
        
        
        figIdx = 2
        resultsFolders = []
        algoLists = []
        tags = [
                    'Sc. 1',
                    'Sc. 2',
                    'Sc. 3',
                ]
        tagsForNonML = None
        args.varyingNumberOfUsersFlag = False
        args.randomPilotsFlag = False
        for loopId, simId in enumerate([3, 4, 5]):
            args.simulationId = simId
            args.scenario = scenarioMapping(simId)
            simulationParameters = SimulationParameters(args)
            systemParameters = SystemParameters(simulationParameters)
            resultsFolders.append(simulationParameters.resultsFolder)
            
            algoList = systemParameters.models
            algoList.append('EPA')
            algoList.append('APG')
            algoLists.append(algoList)
        
        plotFolder = os.path.join(plotFolderBase, f'Fig{figIdx}')
        handleDeletionAndCreation(plotFolder, retain=False)
        consolidatePlot(figIdx, resultsFolders, algoLists, tags, tagsForNonML, plotFolder)
        
        
        figIdx = 3
        resultsFolders = []
        algoLists = []
        tags = [
                    'Sc. 3 with Var K',
                ]
        tagsForNonML = None
        args.varyingNumberOfUsersFlag = True
        args.randomPilotsFlag = False
        for loopId, simId in enumerate([5]):
            args.simulationId = simId
            args.scenario = scenarioMapping(simId)
            simulationParameters = SimulationParameters(args)
            systemParameters = SystemParameters(simulationParameters)
            resultsFolders.append(simulationParameters.resultsFolder)
            
            algoList = systemParameters.models
            algoList.append('EPA')
            algoList.append('APG')
            algoLists.append(algoList)
        
        plotFolder = os.path.join(plotFolderBase, f'Fig{figIdx}')
        handleDeletionAndCreation(plotFolder, retain=False)
        consolidatePlot(figIdx, resultsFolders, algoLists, tags, tagsForNonML, plotFolder)
        
        
        figIdx = 4
        resultsFolders = []
        algoLists = []
        tags = [
                    'Trained on Sc. 2 - Tested on Sc. 2',
                    'Trained on Sc. 3 - Tested on Sc. 2',
                ]
        tagsForNonML =  [
                            'Sc. 2',
                            'Sc. 3',
                        ]
        args.varyingNumberOfUsersFlag = False
        args.randomPilotsFlag = False
        args.minNumberOfUsersFlag = False
        for loopId, simId in enumerate([4, 5]):
            args.simulationId = simId
            if loopId==1:
                args.minNumberOfUsersFlag = True
            
            args.scenario = scenarioMapping(simId)
            simulationParameters = SimulationParameters(args)
            systemParameters = SystemParameters(simulationParameters)
            resultsFolders.append(simulationParameters.resultsFolder)
            
            algoList = systemParameters.models
            if loopId==0:
                algoList.append('EPA')
                algoList.append('APG')
            algoLists.append(algoList)
            tags.append(str(simId))
        
        plotFolder = os.path.join(plotFolderBase, f'Fig{figIdx}')
        handleDeletionAndCreation(plotFolder, retain=False)
        consolidatePlot(figIdx, resultsFolders, algoLists, tags, tagsForNonML, plotFolder)
    else:
        from powerControl.testing import localPlotEditing
        from utils.utils import handleDeletionAndCreation
        
        plotFolderBase = os.path.join(args.root, 'consolidatedResults')
        outputFolderBase = os.path.join(args.root, 'updatedResults')
        handleDeletionAndCreation(outputFolderBase, retain=False)
        args.operatingMode = OperatingModes.PLOTTING_ONLY
        
        for figIdx in range(1, 5):
            plotFolder = os.path.join(plotFolderBase, f'Fig{figIdx}')
            outputFolder = os.path.join(outputFolderBase, f'Fig{figIdx}')
            handleDeletionAndCreation(outputFolder, retain=False)
            localPlotEditing(figIdx, plotFolder, outputFolder)

    # Compute and display the execution time.
    finish = time.perf_counter()
    print(f'Finished in {round(finish - start, 2)} second(s)')