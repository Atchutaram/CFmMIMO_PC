import time
import os

from parameters.modes import OperatingModes
from utils.handleInputArgs import Args
from utils.utils import cleanFolders

defaultNumberOfSamples = 50
testingNumberOfSamples = 50

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

    simulationParameters = SimulationParameters(args)
    systemParameters = SystemParameters(simulationParameters)

    # Generating train & validation or test data.
    if not os.listdir(simulationParameters.dataFolder):
        timeThen = time.perf_counter()
        
        for sampleId in range(args.numberOfSamples):
            # Generates Train/Test data
            dataGen(simulationParameters, systemParameters, sampleId)
            
            if ((simulationParameters.operationMode == OperatingModes.TRAINING)
                and (sampleId < simulationParameters.validationNumberOfData)):
                    # Generates Validation data
                    dataGen(simulationParameters, systemParameters, sampleId, validationData=True)
        
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
        testAndPlot(simulationParameters, systemParameters, plottingOnly=False)

    else:  # simulationParameters.operationMode==OperatingModes.PLOTTING_ONLY
        testAndPlot(simulationParameters, systemParameters, plottingOnly=True)

    # Compute and display the execution time.
    finish = time.perf_counter()
    print(f'Finished in {round(finish - start, 2)} second(s)')