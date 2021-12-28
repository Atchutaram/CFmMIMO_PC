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
    def __init__(self, root, number_of_samples = 20, operation_mode = OperatingModes.training_mode, scenario=1):
        
        self.number_of_samples = number_of_samples
        self.operation_mode = operation_mode
        self.model_folder = f'models_sc_{scenario}'

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.root_path = root
        self.base_folder = 'data_logs_training' if self.operation_mode == OperatingModes.training_mode else 'data_logs_testing'
        self.scenario = scenario
        
        if not os.path.exists(self.root_path):
            print(self.root_path)
            print('root_path failure!')
            sys.exit()
        
        self.base_folder_path = os.path.join(self.root_path, self.base_folder)
        self.model_folder_path = os.path.join(self.root_path, self.model_folder)
        self.data_folder = os.path.join(self.base_folder_path, "betas")
        self.params_folder = os.path.join(self.base_folder_path, "params")
        if not operation_mode==OperatingModes.training_mode:
            self.results_folder = os.path.join(self.base_folder_path, "mus")
        else:
            self.grad_inps_folder = os.path.join(self.base_folder_path, "grad_inps")
            self.grads_folder = os.path.join(self.base_folder_path, "grads")
        
        self.model_file_name = 'model_' + str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '_') + '.pth'

        if not self.operation_mode == OperatingModes.plotting_only:
            if not os.path.exists(self.base_folder_path):
                os.makedirs(self.base_folder_path)  # root_path already exists and verified.

            if not os.path.exists(self.params_folder):
                os.makedirs(self.params_folder)

            handle_deletion_and_creation(self.data_folder)
            if not operation_mode==OperatingModes.training_mode:
                handle_deletion_and_creation(self.results_folder)
            else:
                handle_deletion_and_creation(self.grad_inps_folder)
                handle_deletion_and_creation(self.grads_folder)
        
        if not os.path.exists(self.model_folder_path):
            if self.operation_mode == OperatingModes.data_handling_only or self.operation_mode == OperatingModes.testing_mode:
                print('Train the neural network before testing!')
                sys.exit()
            os.makedirs(self.model_folder_path)  # root_path already exists and verified.