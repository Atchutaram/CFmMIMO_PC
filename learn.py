import sys

from setup_sim_params import get_sim_params
from power_control.learning import train


if __name__ == '__main__':
    
    argv = sys.argv[1:]
    if not argv:
        print("Something went wrong!")
        sys.exit()
    
    sim_filename, model_folder = argv
    simulation_parameters = get_sim_params(sim_filename)

    train(simulation_parameters.device, model_folder)