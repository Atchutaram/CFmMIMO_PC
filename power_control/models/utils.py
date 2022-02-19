import os
import torch
import torch.nn as nn
import importlib
import pickle


from utils.utils import find_the_latest_file


find_import_path = lambda model_name : f"power_control.models.{model_name}_model"

def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


def deploy(model, test_sample, model_name, device):
    import_path = find_import_path(model_name)
    module = importlib.import_module(import_path, ".")  # imports the scenarios
    
    sc = pickle.load(open(module.HyperParameters.sc_path, 'rb'))

    with torch.no_grad():
        test_sample = torch.log(test_sample)

        t_shape = test_sample.shape
        test_sample = test_sample.view((1,-1)).to(device='cpu')
        beta_torch = sc.transform(test_sample)[0]
        beta_torch = torch.tensor(beta_torch, device=device, requires_grad=False, dtype=torch.float32).view(t_shape)
        mus_predicted = model(beta_torch)
        return mus_predicted

def initialize_hyper_params(model_name, simulation_parameters, system_parameters, is_test_mode=False):
    import_path = find_import_path(model_name)
    module = importlib.import_module(import_path, ".")  # imports the scenarios
    
    module.HyperParameters.intialize(simulation_parameters, system_parameters, is_test_mode=is_test_mode)

def load_the_latest_model_and_params_if_exists(model_name, model_folder, interm_folder, system_parameters, grads, device, is_testing=False):  
    import_path = find_import_path(model_name)
    module = importlib.import_module(import_path, ".")  # imports the scenarios
        
    model = module.NeuralNet(device, system_parameters, interm_folder, grads)
    model.apply(initialize_weights)
    model_file = find_the_latest_file(model_folder)
    
    if model_file is not None:
        model.load_state_dict(torch.load(os.path.join(model_folder, model_file)))
    elif is_testing:
        from sys import exit
        print(model_folder)
        print('Train the neural network before testing!')
        exit()
    
    if not is_testing:
        model.to(device)
        model.set_folder(model_folder)
    
    print(model_file)
    return model