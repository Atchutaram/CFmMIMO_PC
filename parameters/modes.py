from enum import IntEnum, auto


class OperatingModes(IntEnum):
    # generates training data and perform training and save trained NN model
    TRAINING  = auto()
    
    # Performs dataGen and test for all the algos for given scenarios.
    # The DL algo uses the trained NN model
    TESTING  = auto()
    
    # This is needs test data. So it should be run at least once before plotting.
    PLOTTING_ONLY = auto()
    
    # Performs training, testing, and then plotting.
    FULL_CHAIN = auto()
    
    # Provides Consolidated plots from all simIds
    CONSOL = auto()
    
    # Fetches Consolidated plots To do further processing
    LOCAL = auto()