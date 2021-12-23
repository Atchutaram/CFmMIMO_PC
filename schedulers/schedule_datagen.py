import sys

from setup_sim_params import get_sim_params
from channel_env.generate_beta import data_gen



if __name__ == '__main__':
    
    argv = sys.argv[1:]
    if not argv:
        print("Something went wrong!")
        sys.exit()
    
    sim_filename, sample_id = argv
    simulation_parameters = get_sim_params(sim_filename)

    data_gen(simulation_parameters.data_folder, sample_id, simulation_parameters.scenario, simulation_parameters.device)