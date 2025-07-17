import argparse
from parameters.modes import OperatingModes
from sys import exit
import os


def composite(value):
    """
    Takes a single parameter as input
    Converts it to an integer and then to a string.

    """

    return str(int(value))

    
def checkNonNegative(value):
    """
    Takes a single parameter as input
    Converts it to an integer if it is a non-negative number else raises an exception.

    """

    intValue = int(value)
    if intValue < 0:
        raise argparse.ArgumentTypeError(f"{value} is an invalid positive int value")
    return intValue


def checkPositive(value):
    """
    Takes a single parameter as input
    Converts it to an integer if it is a non-negative number else raises an exception.

    """

    intValue = int(value)
    if intValue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is an invalid positive int value")
    return intValue


class Args():

    def __init__(self, defaultNumberOfSamples):
    
        parser = argparse.ArgumentParser(
            description=('Train or test the DNN for CFmMIMO downlink power control described in the'
                        ' paper "Pilot Contamination Aware Transformer Neural Network for Downlink '
                        'Power Control in Cell-Free Massive MIMO Networks.')
            )

        parser.add_argument(
                '-id',
                '--simulationId',
                type=checkNonNegative,
                help='All the logs and results folders use this non negative int id. Default 0.',
                default="0",
                metavar='simId', 
            )

        parser.add_argument(
                '-s',
                '--samples',
                type=checkPositive,
                help=('Number of training samples.'
                    'Takes a positive int as input. Valid only for TRAINING phase.'),
                default=defaultNumberOfSamples, metavar='numberOfSamples',
            )

        parser.add_argument(
                '-m',
                '--mode',
                choices=list(map(composite, OperatingModes)),
                help=""" Operating mode.
                    It takes the values from [1-7] to choose one of the following operation modes\n
                    1) TRAINING           : Generates training data and performs training upon all
                    algos.\n
                    2) TESTING            : Generates testing data, performs all the power control
                    algos upon same data, and plots the results.\n
                    3) PLOTTING_ONLY      : Plots the results of a test that is already done.\n
                    4) FULL_CHAIN                : Train, Test, and then Plot.\n
                    5) CONSOL             : This is for generating consolidated plots once all the
                    results of all the simIds are ready.\n
                    6) LOCAL              : To Fetch the consolidated plots and do additional
                    processing like annotation.\n
                    7) INSIGHTS           : To visualize attention matrix during inference for a
                    few realizations with an intentionally designed setting where users having
                    similar channels sharing pilots.\n""",
                default=OperatingModes.FULL_CHAIN,
                metavar='operatingMode',
            )

        parser.add_argument(
                '-sc',
                '--scenario',
                choices={"0", "1", "2", "3"},
                help=('Takes [0-2] as input to pick one of the two scenarios described'
                      'in the paper.'),
                default="0",
                metavar='scenario',
            )

        parser.add_argument(
                '-rp',
                '--randomPilotsFlag',
                choices={"0", "1"},
                help='Choose 1 for random pilots and choose 0 for others.',
                default="0",
                metavar='randomPilotsFlag',
            )

        parser.add_argument(
                '-v',
                '--varK',
                choices={"0", "1"},
                help='Choose 1 for variable K and choose 0 for others.',
                default="0",
                metavar='varK',
            )
        parser.add_argument(
                '-mk',
                '--minK',
                choices={"0", "1"},
                help='Choose 1 for setting K=K_MIN and choose 0 for others.'
                'Choice 1 is Invalid in training phase.',
                default="0",
                metavar='minK',
            )

        parser.add_argument(
                '-ho',
                '--host',
                choices={"0", "1"},
                help='CHOICE 1 IS ONLY FOR THE AUTHORS OF THE CODE! This is for triton.',
                default="0",
                metavar='isTriton',
            )

        parser.add_argument(
                '-re',
                '--retain',
                choices={"0", "1"},
                help=('Choose 1 to retain the input data for training and choose 0 for overwriting'
                      'it.'),
                default="1",
                metavar='retainData',
            )

        parser.add_argument(
                '-c',
                '--clean',
                action='store_true',
                help=('No arguments for this option.'
                    ' This option clears data logs, results, plots, models, lightning_logs and'
                    'sc.pkl. If clean opting is enabled, the other options will be ignored.'),
            )

        args = parser.parse_args()
        
        (
            self.simulationId,
            self.numberOfSamples,
            self.operatingMode,
            self.scenario,
            self.randomPilotsFlag,
            self.varyingNumberOfUsersFlag,
            self.minNumberOfUsersFlag,
            self.host,
            self.retain,
            self.clean
        ) = map(int, (
                args.simulationId,
                args.samples,
                args.mode,
                args.scenario,
                args.randomPilotsFlag,
                args.varK,
                args.minK,
                args.host,
                args.retain,
                args.clean
            )
    )
            
    def preProcessArgs(self, testingNumberOfSamples, insightsNumberOfSamples = 10):
        
        # Translating integers to the element of OperatingModes
        self.operatingMode = list(OperatingModes)[self.operatingMode-1]
        self.fullChainFlag = self.operatingMode == OperatingModes.FULL_CHAIN

        if self.fullChainFlag:
            self.operatingMode = OperatingModes.TRAINING

        # Overwrites input argument 'numberOfSamples' if not 'TRAINING' phase.
        if self.operatingMode == OperatingModes.INSIGHTS:
            self.setNumberOfSamples(insightsNumberOfSamples)
        elif not self.operatingMode == OperatingModes.TRAINING:
            self.setNumberOfSamples(testingNumberOfSamples)
        elif self.minNumberOfUsersFlag:
            print('minK feature is not allowed in modes other than TESTING and PLOTTING_ONLY.'
                   ' Resetting minK to 0.')
            self.minNumberOfUsersFlag = 0

        self.retain = (self.retain==1)  # Translating {0, 1} to {False, True}
        self.randomPilotsFlag = (self.randomPilotsFlag == 1)
        self.varyingNumberOfUsersFlag = (self.varyingNumberOfUsersFlag == 1)
        
    
    def setRootDir(self):
        from utils.utils import handleDeletionAndCreation

        cwd = os.getcwd()
        if self.host == 1:
            # This use case is intended only for the authors of this work.
            import pwd
            userId = pwd.getpwuid(os.getuid())[0]
            currentFolderForTriton = cwd.split('/')[-1]
            rootBase = os.path.join('/tmp', f'hsperfdata_{userId}')
            handleDeletionAndCreation(rootBase, forceRetain=True)

            root = os.path.join('/tmp', f'hsperfdata_{userId}', currentFolderForTriton)
            handleDeletionAndCreation(root, forceRetain=True)

            resultsBase = os.path.join('/scratch', 'work', userId, currentFolderForTriton)
            handleDeletionAndCreation(resultsBase, forceRetain=True)
        else:
            root = cwd
            resultsBase = None
            
        self.root = root
        self.resultsBase = resultsBase
    
    def setNumberOfSamples(self, testingNumberOfSamples):
        self.numberOfSamples = testingNumberOfSamples
        
    def setOperatingMode(self, mode):
        self.operatingMode = mode