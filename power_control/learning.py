from tqdm import tqdm
from torch.profiler import tensorboard_trace_handler
import os
import torch

from utils.utils import delete_folder_contents
from .nn_setup import CommonParameters
from .utils import load_the_latest_model_and_params_if_exists

def training_loops(model, train_loader, opt, num_epochs, profiler):
    for epoch_id in tqdm(range(num_epochs)):
            for beta_torch, beta_original in train_loader:
                utility = model.training_step(beta_torch, beta_original, opt, epoch_id)
                if not profiler is None:
                    profiler.step()
                
            if epoch_id % 10 == 0:
                tqdm.write(f'\nUtility: {-utility.mean().item()}')

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
    train_loader = model.train_dataloader()
    
    if CommonParameters.VARYING_STEP_SIZE:
        opt, _ = model.configure_optimizers()
    else:
        opt = model.configure_optimizers()
    
    PROFILER = False

    if PROFILER:
        trace_folder = os.path.join(model_folder, 'trace') 
        delete_folder_contents(trace_folder)
        with torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1),
                on_trace_ready=tensorboard_trace_handler(trace_folder),
                with_stack=True
                ) as profiler:
                    training_loops(model, train_loader, opt, num_epochs, profiler)
    else:
        training_loops(model, train_loader, opt, num_epochs, None)
    
    model_path = os.path.join(model_folder, simulation_parameters.model_file_name)
    torch.save(model.state_dict(), model_path)
    
    print('Training Done!')