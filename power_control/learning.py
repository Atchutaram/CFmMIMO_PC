from .models.utils import initialize_hyper_params, load_the_latest_model_and_params_if_exists
from .gradient_handler import grads
import torch
from pytorch_lightning import Trainer


def train(simulation_parameters, system_parameters):
    models_list = system_parameters.models_list
    model_folder_dict = simulation_parameters.model_subfolder_path_dict
    
    device = simulation_parameters.device

    for model_name in models_list:
        initialize_hyper_params(model_name, simulation_parameters, system_parameters)
        model = load_the_latest_model_and_params_if_exists(model_name, model_folder_dict[model_name], system_parameters, grads)

        gpus = torch.cuda.device_count()
        print('The number of available/used GPUs: ', gpus)
        trainer = Trainer(gpus=-1, max_epochs=model.num_epochs, accelerator="dp")
        
        trainer.fit(model)
        model.save()