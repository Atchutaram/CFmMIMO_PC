from .models.utils import data_preprocessing, load_the_latest_model_and_params_if_exists
from .gradient_handler import grads


def train(simulation_parameters, system_parameters):
    models_list = system_parameters.models_list
    model_folder_dict = simulation_parameters.model_subfolder_path_dict
    interm_folder_dict = simulation_parameters.interm_subfolder_path_dict
    
    M = system_parameters.number_of_access_points
    K = system_parameters.number_of_users

    
    n_samples = simulation_parameters.number_of_samples
    training_data_path = simulation_parameters.data_folder
    scenario = simulation_parameters.scenario
    device = simulation_parameters.device

    for model_name in models_list:
        data_preprocessing(model_name, M, K, n_samples, training_data_path, scenario)
        model = load_the_latest_model_and_params_if_exists(model_name, model_folder_dict[model_name], interm_folder_dict[model_name], system_parameters, grads, device)
        
        model.train()