import os
import torch
from enum import IntEnum, auto
import datetime
import shutil
import time
import sys


def handle_deletion_and_creation(folder):
    for itr in range(5):
        if not os.path.exists(folder):
            break
        shutil.rmtree(folder, ignore_errors=False, onerror=None)
        time.sleep(0.5)

    if os.path.exists(folder):
        print(f"\n'{folder}' folder was not deleted")
        sys.exit()
    
    os.mkdir(folder)


class OperatingModes(IntEnum):
    training_mode = auto()  # generate training data and perform training in data Handler save trained NN model
    testing_mode = auto()  # Perform dataGen and dataHandler for all the algos for given scenarios. The DL algo uses the trained NN model
    data_handling_only = auto()  # this is only for testing mode. Includes plotting.
    plotting_only = auto()  # this is only after testing mode or data_handling_only


class SimulationParameters:
    def __init__(self, number_of_samples = 20, operation_mode = OperatingModes.training_mode, scenario=1):
        
        self.number_of_samples = number_of_samples
        self.operation_mode = operation_mode
        self.model_folder = 'Models_sc' + scenario

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.root_path = os.path.join('//data.triton.aalto.fi', 'work', 'kochark1', 'CFmMIMO_PC_LS')
        self.base_folder = "data_and_results_sc" + scenario
        self.scenario = scenario
        
        if not os.path.exists(self.root_path):
            print('root_path failure!')
            sys.exit()
        self.model_folder_path = os.path.join(self.root_path, self.model_folder)
        self.model_file_name = 'model_' + str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '_') + '.pth'
        self.base_folder_path = os.path.join(self.root_path, self.base_folder)
        self.common_folder = 'data_and_results_training' if self.operation_mode == OperatingModes.training_mode else 'data_and_results_testing'
        self.common_folder_path = os.path.join(self.base_folder_path, self.common_folder)
        self.data_folder = os.path.join(self.common_folder_path, "dataSet")
        self.results_folder = os.path.join(self.common_folder_path, "results")

        if not self.operation_mode == OperatingModes.plotting_only:
            if not os.path.exists(self.base_folder_path):
                os.makedirs(self.base_folder_path)  # root_path already exists and verified.
            
            handle_deletion_and_creation(self.common_folder_path)
            handle_deletion_and_creation(self.results_folder)
            handle_deletion_and_creation(self.data_folder)
        
        if not os.path.exists(self.model_folder_path):
            if self.operation_mode == OperatingModes.data_handling_only or OperatingModes.testing_mode:
                print('Train the neural network before testing!')
                sys.exit()
            os.makedirs(self.model_folder_path)  # root_path already exists and verified.