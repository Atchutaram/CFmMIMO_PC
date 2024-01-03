from enum import IntEnum, auto


class OperatingModes(IntEnum):
    TRAINING  = auto()  # generates training data and perform training and save trained NN model
    TESTING  = auto()  # Performs dataGen and test for all the algos for given scenarios. The DL algo uses the trained NN model
    PLOTTING_ONLY = auto()  # this is needs test data. So it should be run atleast once before plotting.
    ALL = auto()  # Performs training and then testing.