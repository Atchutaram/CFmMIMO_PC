import os
import torch
from pytorch_lightning import Trainer

from .models.nn_setup import CommonParameters
from .utils import load_the_latest_model_and_params_if_exists


def train(simulation_parameters, system_parameters):
    
    device = simulation_parameters.device
    model_folder = simulation_parameters.model_folder_path
    n_samples = simulation_parameters.number_of_samples
    training_data_path = simulation_parameters.data_folder
    scenario = simulation_parameters.scenario

    M = system_parameters.number_of_access_points
    K = system_parameters.number_of_users

    CommonParameters.data_preprocessing(M, K, n_samples, training_data_path, scenario)
    model = load_the_latest_model_and_params_if_exists(model_folder, device, system_parameters, simulation_parameters.interm_folder)
    num_epochs = CommonParameters.num_epochs
    
    
    if torch.cuda.is_available():
        trainer = Trainer(max_epochs=num_epochs, gpus=1, profiler="simple")
    else:
        trainer = Trainer(max_epochs=num_epochs, profiler="simple")
    
    trainer.fit(model)
    
    model_path = os.path.join(model_folder, simulation_parameters.model_file_name)
    torch.save(model.state_dict(), model_path)
    
    print('Training Done!')