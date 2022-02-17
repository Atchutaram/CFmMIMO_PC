import os
import torch
from enum import IntEnum, auto
import datetime
import sys

from utils.utils import handle_deletion_and_creation



class OperatingModes(IntEnum):
    TRAINING  = auto()  # generates training data and perform training in data Handler save trained NN model
    TESTING  = auto()  # Performs dataGen and dataHandler for all the algos for given scenarios. The DL algo uses the trained NN model
    PLOTTING_ONLY = auto()  # this is only after testing mode



class SimulationParameters:
    def __init__(self, root, number_of_samples, operation_mode, scenario, retain, results_base):
        
        self.number_of_samples = number_of_samples
        self.operation_mode = operation_mode
        self.model_folder = f'models_sc_{scenario}'
        self.scenario = scenario
        
        device_text = "cuda" if OperatingModes.TRAINING and torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_text)
        
        self.root_path = root
        self.base_folder = 'data_logs_training' if self.operation_mode == OperatingModes.TRAINING else 'data_logs_testing'
        
        if not os.path.exists(self.root_path):
            print(self.root_path)
            print('root_path failure!')
            sys.exit()
        
        self.base_folder_path = os.path.join(self.root_path, self.base_folder)
        if results_base is None:
            self.model_folder_path = os.path.join(self.root_path, self.model_folder)
            self.interm_folder = os.path.join(self.root_path, 'interm_models')
        else:
            self.model_folder_path = os.path.join(results_base, self.model_folder)
            self.interm_folder = os.path.join(results_base, 'interm_models')
            
        self.data_folder = os.path.join(self.base_folder_path, "betas")
        if not operation_mode==OperatingModes.TRAINING:
            if results_base is None:
                self.results_folder = os.path.join(self.base_folder_path, "results")
                self.plot_folder = os.path.join(self.base_folder_path, "plots")
            else:
                self.results_folder = os.path.join(results_base, "results")
                self.plot_folder = os.path.join(results_base, "plots")
            
        
        date_str = str(datetime.datetime.now().date()).replace('.', '_').replace('.', '_')
        time_str = str(datetime.datetime.now().time()).replace(':', '_').replace('.', '_')
        self.model_file_name = f'model_{date_str}_{time_str}.pth'

        if not self.operation_mode == OperatingModes.PLOTTING_ONLY:
            if not os.path.exists(self.base_folder_path):
                os.makedirs(self.base_folder_path)

            handle_deletion_and_creation(self.data_folder, self.number_of_samples, retain)
            # The above function deletes and re-creates the folder only if retain=False. If we request different number of data samples than existing, then retain fails.
            
            if not operation_mode==OperatingModes.TRAINING:
                handle_deletion_and_creation(self.results_folder)
                handle_deletion_and_creation(self.plot_folder, force_retain= True)
        else:
            if not os.path.exists(self.results_folder) or len(os.listdir(self.results_folder)) == 0:
                print('Run TESTING mode before running PLOTTING_ONLY mode!')
                sys.exit()
            
            handle_deletion_and_creation(self.plot_folder, force_retain= True)

        
        if not os.path.exists(self.model_folder_path):
            if self.operation_mode == OperatingModes.TESTING:
                print('Train the neural network before testing!')
                sys.exit()
            os.makedirs(self.model_folder_path)
        
        if not os.path.exists(self.interm_folder):
            if self.operation_mode == OperatingModes.TRAINING:
                os.makedirs(self.interm_folder)