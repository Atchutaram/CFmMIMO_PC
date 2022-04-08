import imp
import os
import torch
import torch.nn as nn
import importlib
import pickle


from utils.utils import find_the_latest_file


find_import_path = lambda model_name : f"power_control.models.{model_name}_model"

def initialize_weights(m):
    from .TMN_model import Filter1, Filter2, Filter3
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, Filter1):
        nn.init.kaiming_uniform_(m.weights.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, Filter2):
        nn.init.kaiming_uniform_(m.weights.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, Filter3):
        nn.init.kaiming_uniform_(m.weights.data, 1)
        nn.init.constant_(m.bias.data, 0)


def deploy(model, test_sample, model_name, device, **kwargs):
    import_path = find_import_path(model_name)
    module = importlib.import_module(import_path, ".")  # imports the scenarios
    
    sc = pickle.load(open(module.HyperParameters.sc_path, 'rb'))

    with torch.no_grad():
        if model_name == 'GFT':
            test_sample = model.sqrt_laplace_matrix @ test_sample
        elif model_name == 'TMN':
            system_parameters = kwargs['system_parameters']
            nu_mat = get_nu_tensor(torch.squeeze(test_sample, 0), system_parameters)
            test_sample = torch.log(torch.clamp(nu_mat, min=nu_mat.max()*1e-5))
            test_sample = torch.unsqueeze(test_sample, 0)
            # print(sc.data_min_[0:20])
        else:
            test_sample = torch.log(test_sample)
            

        t_shape = test_sample.shape
        test_sample = test_sample.reshape((1,-1)).to(device='cpu')
        test_sample = sc.transform(test_sample)[0]
        test_sample = torch.tensor(test_sample, device=device, requires_grad=False, dtype=torch.float32).view(t_shape)
        model.eval()
        mus_predicted = model(test_sample)
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

def get_nu_tensor(betas, system_parameters):
    from power_control.utils import compute_vmat
    phi_cross_mat = system_parameters.phi_cross_mat
    phi_cross_mat = phi_cross_mat.to(betas.device)
    v_mat = compute_vmat(betas, system_parameters.zeta_p, system_parameters.T_p, phi_cross_mat)  # Eq (5) b X M X K
    nu_mat = torch.einsum('ik, mi, mk -> mki', phi_cross_mat, (torch.sqrt(v_mat) / betas), betas)  # Eq (14) b X M X K X K

    return nu_mat