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
from powerControl.testing import testAndPlot, visualizeInsights


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
    not (args.operatingMode == OperatingModes.LOCAL) and
    not (args.operatingMode == OperatingModes.INSIGHTS)):
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
            
            if args.fullChainFlag:
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
                ]
        tagsForNonML = None
        args.varyingNumberOfUsersFlag = False
        args.randomPilotsFlag = False
        for loopId, simId in enumerate([3]):
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
                    'Sc. 2',
                ]
        tagsForNonML = None
        args.varyingNumberOfUsersFlag = False
        args.randomPilotsFlag = False
        for loopId, simId in enumerate([4]):
            args.simulationId = simId
            args.scenario = scenarioMapping(simId)
            simulationParameters = SimulationParameters(args)
            systemParameters = SystemParameters(simulationParameters)
            resultsFolders.append(simulationParameters.resultsFolder)
            
            algoList = systemParameters.models
            algoList.append('APG')
            algoList.remove('FCN')
            algoLists.append(algoList)
        
        plotFolder = os.path.join(plotFolderBase, f'Fig{figIdx}')
        handleDeletionAndCreation(plotFolder, retain=False)
        consolidatePlot(figIdx, resultsFolders, algoLists, tags, tagsForNonML, plotFolder)
        
        figIdx = 4
        resultsFolders = []
        algoLists = []
        tags = [
                    'Sc. 3',
                ]
        tagsForNonML = None
        args.varyingNumberOfUsersFlag = False
        args.randomPilotsFlag = False
        for loopId, simId in enumerate([5]):
            args.simulationId = simId
            args.scenario = scenarioMapping(simId)
            simulationParameters = SimulationParameters(args)
            systemParameters = SystemParameters(simulationParameters)
            resultsFolders.append(simulationParameters.resultsFolder)
            
            algoList = systemParameters.models
            algoList.append('APG')
            algoLists.append(algoList)
        
        plotFolder = os.path.join(plotFolderBase, f'Fig{figIdx}')
        handleDeletionAndCreation(plotFolder, retain=False)
        consolidatePlot(figIdx, resultsFolders, algoLists, tags, tagsForNonML, plotFolder)
        
        figIdx = 5
        resultsFolders = []
        algoLists = []
        tags = [
                    'Sc. 3 with Var K',
                    'Trained on Sc. 2 - Tested on Sc. 2',
                    'Trained on Sc. 3 - Tested on Sc. 2',
                ]
        tagsForNonML =  [
                            'Sc. 3 with Var K',
                            'Sc. 2',
                            'Sc. 3',
                        ]

        # Predefine simulation IDs in the desired order
        simIds = [5, 4, 5]

        # Predefine flags for each iteration
        varyingNumberOfUsersFlags = [True, False, False]
        randomPilotsFlags = [False, False, False]
        minNumberOfUsersFlags = [False, False, True]

        for loopId, simId in enumerate(simIds):
            # Set args based on predefined flags
            args.simulationId = simId
            args.varyingNumberOfUsersFlag = varyingNumberOfUsersFlags[loopId]
            args.randomPilotsFlag = randomPilotsFlags[loopId]
            args.minNumberOfUsersFlag = minNumberOfUsersFlags[loopId]

            args.scenario = scenarioMapping(simId)
            simulationParameters = SimulationParameters(args)
            systemParameters = SystemParameters(simulationParameters)
            resultsFolders.append(simulationParameters.resultsFolder)

            # Build algoList
            algoList = systemParameters.models
            if 'FCN' in algoList:
                algoList.remove('FCN')
            if 'PAPCNM' in algoList:
                algoList.remove('PAPCNM')
            if loopId in [0, 1]:
                    algoList.append('APG')  # Only for first part of trained-on-Sc2

            algoLists.append(algoList)

        # Plotting
        plotFolder = os.path.join(plotFolderBase, f'Fig{figIdx}')
        handleDeletionAndCreation(plotFolder, retain=False)
        consolidatePlot(figIdx, resultsFolders, algoLists, tags, tagsForNonML, plotFolder)

    elif args.operatingMode == OperatingModes.LOCAL:
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
    elif args.operatingMode == OperatingModes.INSIGHTS:
        from utils.utils import handleDeletionAndCreation
        from powerControl.testing import consolidatePlot

        insightsFolder = os.path.join(args.root, 'insights')
        handleDeletionAndCreation(insightsFolder, retain=False)

        modelFolder = os.path.join(args.root, 'simID4', 'PAPC')
        if not os.path.exists(modelFolder):
            print(f'{modelFolder} does not exist')

        args.simulationId = 4
        args.scenario = 2
        args.numberOfSamples = 10
        
        ins = [True, True, False]
        packs = [False, True, False]
        for experiment, (forInsights, packFirstTp) in enumerate(zip(ins, packs)):
            baseFolder = os.path.join(insightsFolder, f'Exp{experiment}')
            handleDeletionAndCreation(baseFolder, retain=False)
            simulationParameters = SimulationParameters(args, baseFolder)
            systemParameters = SystemParameters(simulationParameters)

            if not os.listdir(simulationParameters.dataFolder):
                for sampleId in range(args.numberOfSamples):
                    dataGen(simulationParameters, systemParameters, sampleId,
                            forInsights=forInsights, packFirstTp=packFirstTp)
            
            visualizeInsights(simulationParameters, systemParameters)
            
    # Compute and display the execution time.
    finish = time.perf_counter()
    print(f'Finished in {round(finish - start, 2)} second(s)')#